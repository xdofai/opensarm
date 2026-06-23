import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from tqdm import tqdm

from utils.train_utils import set_seed, get_normalizer_from_calculated
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data, _get_vr
from utils.pred_smoother import RegressionConfidenceSmoother
from models.moe_deco_gate_reward_model import RewardTransformer
from models.action_estimator import ActionTransformer
from models.siglip_encoder import FrozenSiglipEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _list_episodes(data_dir: str) -> List[str]:
    """List episode_* directories under data_dir (sorted)."""
    eps = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("episode_")
    ]
    eps.sort()
    return eps


def _build_task_to_class_id(task_list: List[str], class_list: List[str], device: torch.device) -> torch.Tensor:
    """Map each task index to its class index (-1 if unassigned). Same logic as SARM2Workspace._build_hierarchy."""
    temp_mapping: Dict[int, int] = {}
    for class_idx, class_desc in enumerate(class_list):
        raw_tasks = [t.strip() for t in class_desc.split(" or ")]
        seen: List[str] = []
        for t in raw_tasks:
            if t and t not in seen:
                seen.append(t)
        for task_name in seen:
            if task_name in task_list:
                temp_mapping[task_list.index(task_name)] = class_idx
    mapping_list = [temp_mapping.get(i, -1) for i in range(len(task_list))]
    return torch.tensor(mapping_list, device=device)


def _slice_act_pri_inputs(cfg, img_emb: torch.Tensor, state: torch.Tensor):
    """Keep only the n_obs_steps "mid" window so the act_pri model sees its pretraining layout.

    The reward-model frame sequence is laid out as [start] + mid (n_obs_steps) + rewind,
    but act_pri is pretrained with max_rewind_steps=0 (mid window only, fixed length n_obs_steps).
    """
    n_obs_steps = cfg.model.n_obs_steps
    mid_img_emb = img_emb[:, :, 1:n_obs_steps + 1, :]    # (B, N, n_obs_steps, D)
    mid_state = state[:, 1:n_obs_steps + 1, :]            # (B, n_obs_steps, state_dim)
    B = mid_img_emb.shape[0]
    mid_lens = torch.full((B,), n_obs_steps, dtype=torch.int32, device=mid_img_emb.device)
    return mid_img_emb, mid_state, mid_lens


def _align_reward_to_frames(
    data_path: str,
    reward_list: List[float],
    frame_num: int,
    eval_frame_gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align sampled rewards (one per eval_frame_gap) to per-frame rewards using timestamp.npy.

    1) reward_list (delta progress at sampled points) -> progress_samples via cumsum
    2) interpolate progress_samples to all frames using timestamps
    3) per-frame reward = diff(interpolated progress)
    """
    ts_path = Path(data_path) / "timestamp.npy"
    ts = np.load(ts_path)  # (frame_num,)

    if ts.shape[0] != frame_num:
        raise ValueError(f"[{data_path}] timestamp.npy len {ts.shape[0]} != frame_num {frame_num}")

    sample_idx = np.arange(0, frame_num, eval_frame_gap, dtype=np.int64)
    if len(sample_idx) != len(reward_list):
        raise ValueError(
            f"[{data_path}] len(sample_idx)={len(sample_idx)} != len(reward_list)={len(reward_list)}"
        )

    reward_arr = np.asarray(reward_list, dtype=np.float64)
    progress_samples = np.cumsum(reward_arr)

    ts_sample = ts[sample_idx]

    if np.any(np.diff(ts_sample) < 0):
        order = np.argsort(ts_sample)
        progress_full = np.interp(ts, ts_sample[order], progress_samples[order])
    else:
        progress_full = np.interp(ts, ts_sample, progress_samples)

    reward_full = np.empty_like(progress_full, dtype=np.float64)
    reward_full[0] = 0.0
    reward_full[1:] = progress_full[1:] - progress_full[:-1]

    return reward_full.astype(np.float32), progress_full.astype(np.float32)


def _init_inference_objects(cfg, device: torch.device, camera_names: List[str]):
    """Initialize normalizer, SigLIP encoder, MoE reward model, act_pri model, and the
    task->class hierarchy + per-task text features once per process."""
    state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, device)

    VLM_encoder = FrozenSiglipEncoder(cfg.encoders.vision_ckpt, device)
    vis_dim = 768
    txt_dim = 768

    task_list = OmegaConf.to_container(cfg.model.task_list, resolve=True)
    class_list = OmegaConf.to_container(cfg.model.class_list, resolve=True)
    task_to_class_id = _build_task_to_class_id(task_list, class_list, device)
    act_pri_feature = VLM_encoder.encode_text(task_list).to(device)  # (num_tasks, txt_dim)

    reward_model = RewardTransformer(
        d_model=cfg.model.d_model,
        vis_emb_dim=vis_dim,
        text_emb_dim=txt_dim,
        state_dim=cfg.model.state_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        num_cameras=len(camera_names),
        # === Multi-gate MoE ===
        num_gates=cfg.model.num_class,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
        gate_units=cfg.model.moe.gate_units,
        gate_act=cfg.model.moe.gate_act,
        gate_dropout=cfg.model.moe.gate_dropout,
        expert_units=cfg.model.moe.expert_units,
        expert_act=cfg.model.moe.expert_act,
        expert_dropout=cfg.model.moe.expert_dropout,
        lambda_balance=cfg.model.moe.lambda_balance,
        lambda_entropy=cfg.model.moe.lambda_entropy,
        lambda_importance=cfg.model.moe.lambda_importance,
    ).to(device)

    act_pri_model = ActionTransformer(
        d_model=cfg.model.act_pri.d_model,
        vis_emb_dim=vis_dim,
        state_dim=cfg.model.act_pri.state_dim,
        n_layers=cfg.model.act_pri.n_layers,
        n_heads=cfg.model.act_pri.n_heads,
        dropout=cfg.model.act_pri.dropout,
        num_cameras=len(camera_names),
        num_tasks=cfg.model.num_tasks,
        num_classes=cfg.model.num_class,
    ).to(device)

    reward_ckpt = torch.load(Path(cfg.eval.ckpt_path_reward), map_location=device)
    reward_model.load_state_dict(reward_ckpt["model"])
    reward_model.eval()

    act_pri_ckpt = torch.load(Path(cfg.eval.ckpt_path_act_pri), map_location=device)
    act_pri_model.load_state_dict(act_pri_ckpt["model"])
    act_pri_model.eval()

    return state_normalizer, VLM_encoder, reward_model, act_pri_model, act_pri_feature, task_to_class_id


@torch.no_grad()
def _process_one_episode(
    cfg,
    device: torch.device,
    camera_names: List[str],
    state_normalizer,
    VLM_encoder,
    reward_model,
    act_pri_model,
    act_pri_feature,
    task_to_class_id,
    data_path: str,
):
    """Label one episode dir -> reward.npy / progress.npy (per-frame, aligned to timestamps)."""
    ep_name = os.path.basename(data_path)
    frame_num = get_frame_num(data_path)
    traj_joint_data = get_traj_data(data_path)
    eval_frame_gap = cfg.eval.eval_frame_gap

    # With use_future_step the reward head tracks steps-to-go (decreasing), so progress is
    # the negated delta and is smoothed monotonically downward; otherwise it tracks progress.
    use_future_step = cfg.model.use_future_step
    monotonic_mode = "decrease" if use_future_step else "increase"

    reward_list: List[float] = []
    prev_progress = 0.0
    smoother = RegressionConfidenceSmoother(
        value_range=(0.0, 1.0),
        window_size=5,
        beta=2.0,
        monotonic_mode=monotonic_mode,
    )

    lang_strs = [cfg.eval.lang_str]
    lens = torch.tensor([1 + cfg.model.n_obs_steps], dtype=torch.int32, device=device)

    print(f"[proc {device}] start {ep_name} frames={frame_num}")

    for idx in range(0, frame_num, eval_frame_gap):
        batch = get_frame_data_fast(
            path=data_path,
            traj_joint_data=traj_joint_data,
            idx=idx,
            n_obs_steps=cfg.model.n_obs_steps,
            frame_gap=cfg.model.frame_gap,
            max_rewind_steps=cfg.model.max_rewind_steps,
            camera_names=camera_names,
            device=device,
        )

        B, T = batch["image_frames"][camera_names[0]].shape[:2]

        img_list = []
        for cam in camera_names:
            imgs = batch["image_frames"][cam].flatten(0, 1).to(device)  # (B*T, C, H, W)
            img_list.append(imgs)

        state = batch["state"].to(device)
        state = state_normalizer.normalize(state)
        if cfg.model.no_state:
            state = torch.zeros_like(state, device=device)

        # SigLIP encoding
        imgs_all = torch.cat(img_list, dim=0)                              # (N * B * T, C, H, W)
        img_emb = VLM_encoder.encode_image(imgs_all)                       # (N * B * T, 768)
        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, 768)
        task_inst_emb = VLM_encoder.encode_text(lang_strs)                 # (B, 768)

        # act_pri sees only the mid window (no start frame, no rewind)
        act_pri_img_emb, act_pri_state, act_pri_lens = _slice_act_pri_inputs(cfg, img_emb, state)
        act_pri_pred, _ = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)  # (B, num_tasks)
        act_pri_pred_idx = torch.argmax(act_pri_pred, dim=-1)              # (B,)
        lang_emb = act_pri_feature[act_pri_pred_idx].to(device)           # (B, 768)
        gate_idx = task_to_class_id[act_pri_pred_idx]                     # (B,) MoE gate (class) index

        reward_pred, _, _ = reward_model(img_emb, lang_emb, task_inst_emb, state, lens, gate_idx=gate_idx)  # (B, T)
        pred = torch.clip(reward_pred, 0, 1)
        raw_item = pred[0, cfg.model.n_obs_steps].item()

        conf_val = 1.0  # placeholder (MoE reward model exposes no per-step confidence)
        smoothed_item = smoother.update(raw_item, conf_val)

        reward = smoothed_item - prev_progress
        if use_future_step:
            reward = -reward
        reward_list.append(reward)
        prev_progress = smoothed_item

    reward_full, progress_full = _align_reward_to_frames(
        data_path=data_path,
        reward_list=reward_list,
        frame_num=frame_num,
        eval_frame_gap=eval_frame_gap,
    )

    if cfg.eval.get("clip_reward_nonneg", False):
        reward_full = np.maximum(reward_full, 0.0).astype(np.float32)

    np.save(Path(data_path) / "reward.npy", reward_full)
    np.save(Path(data_path) / "progress.npy", progress_full)

    print(f"[proc {device}] done {ep_name} -> reward.npy (per-frame)")


def label_reward_worker_entry(
    rank: int,
    world_size: int,
    cfg_dict: Dict[str, Any],
    episodes: List[str],
):
    """Spawn entry (MUST be module-level). Each worker binds to one GPU and shards episodes."""
    cfg = OmegaConf.create(cfg_dict)

    assert torch.cuda.is_available(), "CUDA required"
    ngpu = torch.cuda.device_count()
    device_id = rank % ngpu
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    camera_names = list(cfg.general.camera_names)

    (state_normalizer, VLM_encoder, reward_model, act_pri_model,
     act_pri_feature, task_to_class_id) = _init_inference_objects(cfg, device, camera_names)

    my_eps = episodes[rank::world_size]
    print(f"[rank={rank}] device={device} episodes={len(my_eps)}")

    for ep in tqdm(my_eps, desc=f"[rank={rank}] Processing episodes"):
        out_path = Path(ep) / "reward.npy"
        if out_path.exists():
            print(f"[rank={rank}] skip {os.path.basename(ep)} (reward.npy exists)")
            continue

        _process_one_episode(
            cfg=cfg,
            device=device,
            camera_names=camera_names,
            state_normalizer=state_normalizer,
            VLM_encoder=VLM_encoder,
            reward_model=reward_model,
            act_pri_model=act_pri_model,
            act_pri_feature=act_pri_feature,
            task_to_class_id=task_to_class_id,
            data_path=ep,
        )
        _get_vr.cache_clear()


class SARM2LabelWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names

    def clear_labels(self):
        data_dir = self.cfg.eval.label_data_dir
        episodes = _list_episodes(data_dir)
        removed = 0
        for ep in episodes:
            for fname in ("reward.npy", "progress.npy"):
                p = Path(ep) / fname
                if p.exists():
                    p.unlink()
                    removed += 1
        print(f"[Clear] removed {removed} files across {len(episodes)} episodes in {data_dir}")

    def label_reward(self):
        cfg = self.cfg
        camera_names = list(self.camera_names)

        (state_normalizer, VLM_encoder, reward_model, act_pri_model,
         act_pri_feature, task_to_class_id) = _init_inference_objects(cfg, self.device, camera_names)

        data_dir = cfg.eval.label_data_dir
        all_episodes = _list_episodes(data_dir)
        print(f"[Label] Found {len(all_episodes)} episodes in {data_dir}")

        for i, data_path in enumerate(tqdm(all_episodes, desc="[Label] Processing episodes")):
            out_path = Path(data_path) / "reward.npy"
            if out_path.exists():
                print(f"[Label] skip {os.path.basename(data_path)} (reward.npy exists)")
                continue

            print(f"[Label] {i+1}/{len(all_episodes)} {os.path.basename(data_path)}")
            _process_one_episode(
                cfg=cfg,
                device=self.device,
                camera_names=camera_names,
                state_normalizer=state_normalizer,
                VLM_encoder=VLM_encoder,
                reward_model=reward_model,
                act_pri_model=act_pri_model,
                act_pri_feature=act_pri_feature,
                task_to_class_id=task_to_class_id,
                data_path=data_path,
            )
            _get_vr.cache_clear()

    def label_reward_multi_process(self):
        import torch.multiprocessing as mp

        cfg = self.cfg
        data_dir = cfg.eval.label_data_dir

        episodes = _list_episodes(data_dir)
        if len(episodes) == 0:
            print(f"[WARN] No episodes found in: {data_dir}")
            return

        assert torch.cuda.is_available(), "CUDA required for multi-process labeling"
        ngpu = torch.cuda.device_count()
        world_size = min(2, ngpu)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        mp.set_start_method("spawn", force=True)
        mp.spawn(
            label_reward_worker_entry,
            args=(world_size, cfg_dict, episodes),
            nprocs=world_size,
            join=True,
        )
