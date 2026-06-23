import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.train_utils import set_seed, get_normalizer_from_calculated
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data, normalize_sparse, normalize_dense, _get_vr
from utils.pred_smoother import RegressionConfidenceSmoother
from models.subtask_estimator import SubtaskTransformer
from models.stage_estimator import StageTransformer
from models.clip_encoder import FrozenCLIPEncoder

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
    """Initialize normalizer, CLIP encoder, subtask model, and stage model once per process."""
    state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, device)
    clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, device)
    vis_dim, txt_dim = 512, 512

    subtask_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.subtask_model
    stage_model_path = Path(cfg.eval.ckpt_path) / cfg.eval.stage_model

    model_type = cfg.eval.model_type
    if model_type == "sparse":
        num_classes = cfg.model.num_classes_sparse
    else:
        num_classes = cfg.model.num_classes_dense

    subtask_model = SubtaskTransformer(
        d_model=cfg.model.d_model,
        vis_emb_dim=vis_dim,
        text_emb_dim=txt_dim,
        state_dim=cfg.model.state_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        num_cameras=len(camera_names),
    ).to(device)

    stage_model = StageTransformer(
        d_model=cfg.model.d_model,
        vis_emb_dim=vis_dim,
        text_emb_dim=txt_dim,
        state_dim=cfg.model.state_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        num_cameras=len(camera_names),
        num_classes_sparse=cfg.model.num_classes_sparse,
        num_classes_dense=cfg.model.num_classes_dense,
    ).to(device)

    subtask_ckpt = torch.load(subtask_model_path, map_location=device)
    stage_ckpt = torch.load(stage_model_path, map_location=device)
    subtask_model.load_state_dict(subtask_ckpt["model"])
    stage_model.load_state_dict(stage_ckpt["model"])

    subtask_model.eval()
    stage_model.eval()

    return state_normalizer, clip_encoder, subtask_model, stage_model, model_type, num_classes


@torch.no_grad()
def _process_one_episode(
    cfg,
    device: torch.device,
    camera_names: List[str],
    state_normalizer,
    clip_encoder,
    subtask_model,
    stage_model,
    model_type: str,
    num_classes: int,
    data_path: str,
):
    """Label one episode dir -> reward.npy / progress.npy (per-frame, aligned to timestamps)."""
    ep_name = os.path.basename(data_path)
    frame_num = get_frame_num(data_path)
    traj_joint_data = get_traj_data(data_path)
    eval_frame_gap = cfg.eval.eval_frame_gap

    reward_list: List[float] = []
    prev_progress = 0.0
    smoother = RegressionConfidenceSmoother(value_range=(0.0, 1.0))

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

        # CLIP encoding
        imgs_all = torch.cat(img_list, dim=0)                              # (N * B * T, C, H, W)
        img_emb = clip_encoder.encode_image(imgs_all)                      # (N * B * T, 512)
        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, 512)
        lang_emb = clip_encoder.encode_text(lang_strs)                     # (B, 512)

        # Stage model -> stage index + confidence
        stage_prob = stage_model(img_emb, lang_emb, state, lens, scheme=model_type).softmax(dim=-1)  # (B, T, C)
        stage_idx = stage_prob.argmax(dim=-1)                              # (B, T)
        stage_conf = stage_prob.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Subtask model with stage prior
        stage_onehot = F.one_hot(stage_idx, num_classes=stage_prob.size(-1)).float()  # (B, T, C)
        stage_emb = stage_onehot.unsqueeze(1)                              # (B, 1, T, C)
        subtask_pred = subtask_model(img_emb, lang_emb, state, lens, stage_emb)
        pred = torch.clip(subtask_pred + stage_idx.float(), 0, num_classes - 1)  # (B, T)

        raw_item = pred[0, cfg.model.n_obs_steps].item()
        if model_type == "sparse":
            raw_item_norm = normalize_sparse(raw_item)
        else:
            raw_item_norm = normalize_dense(raw_item)

        conf_val = stage_conf[0, cfg.model.n_obs_steps].item()
        smoothed_item = smoother.update(raw_item_norm, conf_val)

        reward = smoothed_item - prev_progress
        reward_list.append(reward)
        prev_progress = smoothed_item

    reward_full, progress_full = _align_reward_to_frames(
        data_path=data_path,
        reward_list=reward_list,
        frame_num=frame_num,
        eval_frame_gap=eval_frame_gap,
    )

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
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg_dict)

    assert torch.cuda.is_available(), "CUDA required"
    ngpu = torch.cuda.device_count()
    device_id = rank % ngpu
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    camera_names = list(cfg.general.camera_names)

    (state_normalizer, clip_encoder, subtask_model, stage_model,
     model_type, num_classes) = _init_inference_objects(cfg, device, camera_names)

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
            clip_encoder=clip_encoder,
            subtask_model=subtask_model,
            stage_model=stage_model,
            model_type=model_type,
            num_classes=num_classes,
            data_path=ep,
        )
        _get_vr.cache_clear()


class SARMLabelWorkspace:
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

        (state_normalizer, clip_encoder, subtask_model, stage_model,
         model_type, num_classes) = _init_inference_objects(cfg, self.device, camera_names)

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
                clip_encoder=clip_encoder,
                subtask_model=subtask_model,
                stage_model=stage_model,
                model_type=model_type,
                num_classes=num_classes,
                data_path=data_path,
            )
            _get_vr.cache_clear()

    def label_reward_multi_process(self):
        import torch.multiprocessing as mp
        
        from omegaconf import OmegaConf

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
