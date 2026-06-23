import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from tqdm import tqdm
import wandb

from lerobot.common.datasets.rm_act_pri_lerobot_dataset import FrameGapLeRobotDataset 
from utils.data_utils import get_valid_episodes, split_train_eval_episodes, adapt_lerobot_batch_act_pri
from utils.train_utils import set_seed, save_ckpt, get_normalizer_from_calculated, plot_episode_result, plot_episode_result_raw_data, plot_act_pri_result
from utils.raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data
from utils.make_demo_video import produce_video
from models.moe_deco_gate_reward_model import RewardTransformer
from models.action_estimator import ActionTransformer
from models.siglip_encoder import FrozenSiglipEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"


class SARM2Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.task_list = OmegaConf.to_container(cfg.model.task_list, resolve=True)
        self.class_list = OmegaConf.to_container(cfg.model.class_list, resolve=True)
        self.class_task_dict = {}
        self._build_hierarchy()

    def _build_hierarchy(self):
        # Temporary mapping to help build the tensor later
        # Maps task_index -> class_index
        temp_mapping = {}

        for class_idx, class_desc in enumerate(self.class_list):
            # Split, clean, and deduplicate tasks in the class string
            raw_tasks = [t.strip() for t in class_desc.split(" or ")]
            
            unique_tasks_in_class = []
            for t in raw_tasks:
                if t and t not in unique_tasks_in_class:
                    unique_tasks_in_class.append(t)
            
            self.class_task_dict[class_desc] = {}
            
            for task_name in unique_tasks_in_class:
                if task_name in self.task_list:
                    task_idx = self.task_list.index(task_name)
                    # Store task index in the dictionary
                    self.class_task_dict[class_desc][task_name] = task_idx
                    # Record which class this task index belongs to
                    temp_mapping[task_idx] = class_idx
                else:
                    # Skips tasks like 'install' or 'assembly' if not in task_list
                    continue

        # Construct the TASK_TO_CLASS_ID tensor for the loss function
        # Defaults to -1 if a task is somehow not assigned to a class
        mapping_list = [temp_mapping.get(i, -1) for i in range(len(self.task_list))]
        self.task_to_class_id = torch.tensor(mapping_list, device=self.device)
        print("[Init] Task to Class ID mapping")

    def _slice_act_pri_inputs(self, img_emb, state):
        """Align the reward-model sequence to the act_pri pretraining layout.

        IMPORTANT (no-rewind assumption):
        The act_pri model (act_pri_ws.py + config/act_pri.yaml) is pretrained with
        `stage_model=True` and `max_rewind_steps=0`, so its input sequence is ONLY the
        `n_obs_steps` "mid" frames (no episode-start frame, no rewind frames) and its
        sequence length is ALWAYS exactly `n_obs_steps`.

        The reward-model dataset here instead builds a longer sequence laid out along the
        time axis as:
            [start] + mid (n_obs_steps frames) + rewind (max_rewind_steps frames)
        i.e. index 0 is the episode-start frame, indices 1..n_obs_steps are the mid frames,
        and the trailing indices are rewind frames. (This matches
        adapt_lerobot_batch_act_pri, which reads act_pri_index[:, 1:n_obs_steps+1].)

        Feeding that full sequence to act_pri is a train/inference mismatch. So we keep only
        the mid window [1 : n_obs_steps+1] and hand act_pri a fixed length of `n_obs_steps`
        for every sample -- we assume act_pri never rewinds, so no per-sample `lens` is used.
        """
        n_obs_steps = self.cfg.model.n_obs_steps
        mid_img_emb = img_emb[:, :, 1:n_obs_steps + 1, :]   # (B, N, n_obs_steps, D)
        mid_state = state[:, 1:n_obs_steps + 1, :]           # (B, n_obs_steps, state_dim)
        B = mid_img_emb.shape[0]
        # no-rewind assumption: act_pri length is always n_obs_steps (the full mid window)
        mid_lens = torch.full((B,), n_obs_steps, dtype=torch.int32, device=mid_img_emb.device)
        return mid_img_emb, mid_state, mid_lens

    def train(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")
        cfg = self.cfg
        OmegaConf.save(cfg, self.save_dir / "config.yaml")
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}',
            name=f'{cfg.general.task_name}/{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}/stepsToGo-{cfg.model.use_future_step}',
            config=cfg,
        )

        # --- data ---
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        dataset_train = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=train_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               no_pertube=cfg.model.no_pertube,
                                               task_list=cfg.model.task_inst_list, # use whole task instruction list for training
                                               pre_decode_video_frames=cfg.model.pre_decode_video_frames,
                                               )
        

        dataset_val = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               no_pertube=cfg.model.no_pertube,
                                               task_list=cfg.model.task_inst_list,
                                               pre_decode_video_frames=cfg.model.pre_decode_video_frames,)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)


        # SigLIP encoder
        VLM_encoder = FrozenSiglipEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 768; txt_dim = 768
        
        
        self.act_pri_feature = VLM_encoder.encode_text(self.task_list).to(self.device)  # (num_tasks, txt_dim)
        

        
        # --- reward_model ---
       
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                   # === Multi-gate MoE ===
                                  num_gates=cfg.model.num_class,
                                   # === MoE hyper-parameters ===
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
                                  ).to(self.device)
        
        act_pri_model = ActionTransformer(d_model=cfg.model.act_pri.d_model, 
                                    vis_emb_dim=vis_dim, 
                                    state_dim=cfg.model.act_pri.state_dim,
                                    n_layers=cfg.model.act_pri.n_layers,
                                    n_heads=cfg.model.act_pri.n_heads,
                                    dropout=cfg.model.act_pri.dropout,
                                    num_cameras=len(self.camera_names),
                                    num_tasks=cfg.model.num_tasks,
                                    num_classes=cfg.model.num_class,
                                  ).to(self.device)
        
        if cfg.model.resume_training:
            reward_model_path = Path(cfg.model.model_path)
            # Load checkpoints
            reward_ckpt = torch.load(reward_model_path, map_location=self.device)
            # Load weights
            reward_model.load_state_dict(reward_ckpt["model"])
            # Move to device
            reward_model.to(self.device)
            reward_model.train()
            print(f"[Init] Resumed training from {reward_model_path} and {act_pri_model_path}")


        # act_pri_model: load pretrained, optionally finetune
        finetune_act_pri = getattr(cfg.model, 'finetune_act_pri', False)
        act_pri_loss_weight = getattr(cfg.model, 'act_pri_loss_weight', 1.0)
        act_pri_model_path = Path(cfg.model.act_pri_model_path)
        act_pri_ckpt = torch.load(act_pri_model_path, map_location=self.device)
        act_pri_model.load_state_dict(act_pri_ckpt["model"])
        act_pri_model.to(self.device)
        if not finetune_act_pri:
            for p in act_pri_model.parameters():
                p.requires_grad_(False)
            act_pri_model.eval()

        # Optimizer
        trainable_params = list(reward_model.parameters())
        if finetune_act_pri:
            trainable_params += list(act_pri_model.parameters())
        reward_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        self.alpha_auxiliary = cfg.model.moe.alpha_auxiliary
        
        
        # Schedulers
        # Reward scheduler
        reward_warmup_scheduler = LinearLR(
            reward_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  # or 0.0 for full ramp-up
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        reward_cosine_scheduler = CosineAnnealingLR(
            reward_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,  # cosine decay after warmup
            eta_min=0.0  # or set a nonzero final LR if needed
        )
        reward_scheduler = SequentialLR(
            reward_optimizer,
            schedulers=[reward_warmup_scheduler, reward_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        
        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            reward_model.train()
            if finetune_act_pri:
                act_pri_model.train()
            # training stops at the smaller of num_steps and one epoch of batches
            pbar_total = min(cfg.train.num_steps, len(dataloader_train))
            with tqdm(dataloader_train, desc=f"Epoch {epoch}", total=pbar_total) as pbar:
                for batch in pbar:
                    batch = adapt_lerobot_batch_act_pri(batch, n_obs_steps=cfg.model.n_obs_steps, camera_names=cfg.general.camera_names)
                    B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                    img_list = []
                    for key in self.camera_names:
                        imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                        img_list.append(imgs)
                    
                    lang_strs = batch["tasks"]
                    trg = batch["targets"].to(self.device)
                    steps_to_go = batch["steps_to_go"].to(self.device)
                    lens = batch["lengths"].to(self.device)
                    state = batch["state"].to(self.device)
                    act_pri_idx_gt = batch["act_pri_index"].to(self.device)  # (B, T)
                    act_pri_idx_gt = act_pri_idx_gt[:, -1].long() 
                    class_idx_gt = self.task_to_class_id[act_pri_idx_gt]

                    
                    if cfg.model.use_future_step:
                        trg = steps_to_go
                    
                    
                    with torch.no_grad():
                        state = state_normalizer.normalize(state)
                        # VLM encoding
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = VLM_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        task_inst_emb = VLM_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                        if cfg.model.no_state:
                            state = torch.zeros_like(state, device=self.device)

                    # act_pri sees only the mid window (no start frame, no rewind); see _slice_act_pri_inputs
                    act_pri_img_emb, act_pri_state, act_pri_lens = self._slice_act_pri_inputs(img_emb, state)

                    # finetune act_pri estimator
                    if finetune_act_pri:
                        act_pri_pred, class_pred = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)  # (B, num_tasks)
                        act_pri_pred_idx = torch.argmax(act_pri_pred, dim=-1)  # (B,)
                        act_pri_ce_loss = F.cross_entropy(
                            act_pri_pred,
                            act_pri_idx_gt,
                            reduction="mean"
                        )
                        class_ce_loss = F.cross_entropy(
                            class_pred,
                            class_idx_gt,
                            reduction="mean"
                        )
                        # Soft combination to keep path differentiable
                        act_pri_prob = F.softmax(act_pri_pred, dim=-1)  # (B, num_tasks)
                        lang_emb = act_pri_prob @ self.act_pri_feature  # (B, txt_dim)
                    else:
                        with torch.no_grad():
                            act_pri_pred, class_pred = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)  # (B, num_tasks)
                            act_pri_pred_idx = torch.argmax(act_pri_pred, dim=-1)  # (B,)
                            lang_emb = self.act_pri_feature[act_pri_pred_idx].to(self.device)

                    gate_idx = self.task_to_class_id[act_pri_pred_idx] # (B,) class index for MoE gate

                    reward_pred, aux_loss, info = reward_model(img_emb, lang_emb, task_inst_emb, state, lens, gate_idx=gate_idx)
                    fit_loss = F.mse_loss(reward_pred, trg, reduction="mean")
                    reward_loss = fit_loss + self.alpha_auxiliary * aux_loss
                    if finetune_act_pri:
                        total_loss = reward_loss + act_pri_loss_weight * (act_pri_ce_loss + class_ce_loss)
                    else:
                        total_loss = reward_loss


                    reward_optimizer.zero_grad()
                    total_loss.backward()
                    reward_unclipped = nn.utils.clip_grad_norm_(trainable_params, float("inf")).item()
                    _ = nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
                    reward_optimizer.step()
                    reward_scheduler.step()

                    
                    if step % cfg.train.log_every == 0:
                        log_dict = {
                            "train/total_loss": total_loss.item(),
                            "train/fit_loss": fit_loss.item(),
                            "train/lr": reward_scheduler.get_last_lr()[0],
                            "train/reward_grad_norm": reward_unclipped,
                            "train/moe/auxiliary_loss": aux_loss.item(),
                            "train/moe/gate_entropy": info["gate_entropy"],
                            "epoch": epoch,
                        }
                        if finetune_act_pri:
                            log_dict["train/act_pri_ce_loss"] = act_pri_ce_loss.item()
                        wandb.log(log_dict, step=step)

                    pbar.set_postfix(loss=f"{(total_loss.item()):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name=f"reward_step_{step:06d}_loss_{reward_loss.item():.3f}")
                        if finetune_act_pri:
                            save_ckpt(act_pri_model, reward_optimizer, epoch, self.save_dir, input_name=f"act_pri_step_{step:06d}_loss_{act_pri_ce_loss.item():.3f}")
                    step += 1
                    del batch
                    if step >= cfg.train.num_steps:
                        print(f"Reached max training steps: {cfg.train.num_steps}. Stopping training.")
                        break

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                reward_model.eval()
                act_pri_model.eval()
                total_loss, total_act_pri_loss, num = 0.0, 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    with tqdm(dataloader_val, desc=f"Epoch {epoch}") as pbar_val:
                        for batch in pbar_val:
                            batch = adapt_lerobot_batch_act_pri(batch, n_obs_steps=cfg.model.n_obs_steps, camera_names=cfg.general.camera_names)
                            B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                            img_list = []
                            for key in self.camera_names:
                                imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                                img_list.append(imgs)
                            
                            lang_strs = batch["tasks"] # (T, B)
                            trg = batch["targets"].to(self.device)
                            steps_to_go = batch["steps_to_go"].to(self.device)
                            lens = batch["lengths"].to(self.device)
                            state = batch["state"].to(self.device)
                            state = state_normalizer.normalize(state)
                            act_pri_index = batch["act_pri_index"].to(self.device)  # (B, T)
                            
                            
                            if cfg.model.use_future_step:
                                trg = steps_to_go

                            # VLM encoding
                            imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                            img_emb = VLM_encoder.encode_image(imgs_all)  # (N * B * T, D)
                            img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                            task_inst_emb = VLM_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)
                            
                            if cfg.model.no_state:
                                state = torch.zeros_like(state, device=self.device)

                            # act_pri sees only the mid window (no start frame, no rewind); see _slice_act_pri_inputs
                            act_pri_img_emb, act_pri_state, act_pri_lens = self._slice_act_pri_inputs(img_emb, state)
                            act_pri_pred, class_pred = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)

                            # # Scheduled Sampling Logic
                            act_pri_pred_idx = torch.argmax(act_pri_pred, dim=-1)  # (B,)
                            lang_emb = self.act_pri_feature[act_pri_pred_idx].to(self.device)
                            gate_idx = self.task_to_class_id[act_pri_pred_idx]

                            reward_pred, _, info = reward_model(img_emb, lang_emb, task_inst_emb, state, lens, gate_idx=gate_idx)
                            reward_loss = F.mse_loss(reward_pred, trg, reduction="mean")
                            total_loss += reward_loss.item()
                            total_act_pri_loss += F.cross_entropy(act_pri_pred, act_pri_index[:, 0].long()).item()
                            num += 1
                            del batch

                val_loss = total_loss / num
                val_act_pri_loss = total_act_pri_loss / num
                print(f"[Eval] Epoch {epoch} Val MSE: {val_loss:.6f} | Act Pri CE: {val_act_pri_loss:.6f}")
                wandb.log({"val/loss": val_loss, "val/act_pri_ce_loss": val_act_pri_loss}, step=step)
            # --- clear memory ---
            torch.cuda.empty_cache()

            # --- save checkpoints ---
            save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_latest")
            if finetune_act_pri:
                save_ckpt(act_pri_model, reward_optimizer, epoch, self.save_dir, input_name="act_pri_latest")

            if epoch == cfg.train.num_epochs:
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_final")
                if finetune_act_pri:
                    save_ckpt(act_pri_model, reward_optimizer, epoch, self.save_dir, input_name="act_pri_final")

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_best")
                if finetune_act_pri:
                    save_ckpt(act_pri_model, reward_optimizer, epoch, self.save_dir, input_name="act_pri_best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()


    @torch.no_grad()
    def eval(self):
        import random
        from utils.data_utils import find_eps_each_task
        from utils.moe_utils import compute_moe, collect_task_expert_topk_to_csv
        from utils.data_utils import compute_mse, collect_task_mse_to_csv, compute_act_pri_accuracy, collect_task_act_pri_accuracy_to_csv
        
        cfg = self.cfg
        repo_id = cfg.general.repo_id
        self.save_dir = Path(f'{cfg.general.project_name}/eval/{cfg.general.task_name}')
        valid_episodes = get_valid_episodes(repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        
        dataset_val = FrameGapLeRobotDataset(repo_id=repo_id, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               video_eval=True,
                                               no_pertube=True,
                                               task_list=cfg.model.task_inst_list)
        
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # SigLIP encoder
        VLM_encoder = FrozenSiglipEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 768; txt_dim = 768
        
        self.act_pri_feature = VLM_encoder.encode_text(self.task_list).to(self.device)  # (num_tasks, txt_dim)
        
        # Create model instances
        # --- reward_model ---
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                  # === Multi-gate MoE ===
                                  num_gates=cfg.model.num_class,
                                   # === MoE hyper-parameters ===
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
                                  ).to(self.device)
        
        act_pri_model = ActionTransformer(d_model=cfg.model.act_pri.d_model, 
                                    vis_emb_dim=vis_dim, 
                                    state_dim=cfg.model.act_pri.state_dim,
                                    n_layers=cfg.model.act_pri.n_layers,
                                    n_heads=cfg.model.act_pri.n_heads,
                                    dropout=cfg.model.act_pri.dropout,
                                    num_cameras=len(self.camera_names),
                                    num_tasks=cfg.model.num_tasks,
                                    num_classes=cfg.model.num_class,
                                  ).to(self.device)


        # Load checkpoints
        reward_model_path = Path(cfg.eval.ckpt_path_reward) 
        act_pri_model_path = Path(cfg.eval.ckpt_path_act_pri) 
        
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        reward_model.load_state_dict(reward_ckpt["model"])
        reward_model.to(self.device)
        reward_model.eval()
        
        act_pri_ckpt = torch.load(act_pri_model_path, map_location=self.device)
        act_pri_model.load_state_dict(act_pri_ckpt["model"])
        act_pri_model.to(self.device)
        act_pri_model.eval()

        # save path
        rollout_save_dir = Path(self.save_dir) / "eval_video"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        evaled_list = []
        split_task = cfg.eval.split_task
        act_pri_loss_task = 0.0
        act_pri_loss_total = 0.0

        
        if not split_task:
            for i in range(cfg.eval.run_times):
                ep_index = random.choice([idx for idx in val_eps if idx not in evaled_list])
                act_pri_loss_ep = self._eval_one_episode(ep_index, i+1, val_eps, evaled_list, dataset_val, state_normalizer, VLM_encoder, act_pri_model, reward_model, split_task, rollout_save_dir, repo_id)
                act_pri_loss_task += act_pri_loss_ep
                act_pri_loss_total += act_pri_loss_ep
        else:
            count = 1
            act_pri_loss_task = 0.0
            whole_task_list = OmegaConf.to_container(cfg.model.task_inst_list, resolve=True)  
            num_per_task = cfg.eval.run_times // len(whole_task_list)
            ep_dict = find_eps_each_task(num_per_task, val_eps, repo_id, whole_task_list, cfg.general.seed)
            for task in whole_task_list:
                for ep_index in ep_dict[task]:
                    act_pri_loss_ep = self._eval_one_episode(ep_index, count, val_eps, evaled_list, dataset_val, state_normalizer, VLM_encoder, act_pri_model, reward_model, split_task, rollout_save_dir, repo_id)
                    act_pri_loss_task += act_pri_loss_ep
                    act_pri_loss_total += act_pri_loss_ep
                    count += 1
                task_dir = rollout_save_dir / task
                compute_moe(task_dir, top_k=cfg.model.moe.top_k)
                compute_mse(task_dir)
                compute_act_pri_accuracy(task_dir,
                                         task_to_class_id=self.task_to_class_id,
                                         dummy_task_idx=len(self.task_list) - 1)
            
            
            compute_moe(rollout_save_dir, top_k=cfg.model.moe.top_k)
            compute_mse(rollout_save_dir)
            compute_act_pri_accuracy(rollout_save_dir,
                                     task_to_class_id=self.task_to_class_id,
                                     dummy_task_idx=len(self.task_list) - 1)
            collect_task_expert_topk_to_csv(rollout_save_dir, top_k=cfg.model.moe.top_k)
            collect_task_mse_to_csv(rollout_save_dir)
            collect_task_act_pri_accuracy_to_csv(rollout_save_dir)
        
                

    def _eval_one_episode(self, ep_index, count, val_eps, evaled_list, dataset_val, state_normalizer, VLM_encoder, act_pri_model, reward_model, split_task, rollout_save_dir, repo_id):
            from utils.train_utils import generate_whole_task_reward
            cfg = self.cfg
            global_idx = val_eps.index(ep_index)
            evaled_list.append(ep_index)
            start_idx = dataset_val.episode_data_index["from"][global_idx].item()
            end_idx = dataset_val.episode_data_index["to"][global_idx].item() - 1
            gt_ep_result = []
            pred_ep_result = []
            pred_ep_smoothed = []
            gt_act_pri_result = []
            pred_act_pri_result = []
            pred_act_pri_conf = []
            aux_loss_result = []
            expert_load_result = []
            x_offset = 0
            # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
            eval_frame_gap = cfg.eval.eval_frame_gap
            print(f"[Eval Video] Evaluating episode_{ep_index}, progress: {count} / {cfg.eval.run_times}")

            # change to use tqdm
            for idx in tqdm(range(start_idx, end_idx, eval_frame_gap), desc=f"Processing episode {ep_index}"):
                data_point = dataset_val[idx]
                batch = adapt_lerobot_batch_act_pri(
                    data_point, 
                    camera_names=cfg.general.camera_names, 
                    eval_video=True,
                    n_obs_steps = cfg.model.n_obs_steps
                )
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                steps_to_go = batch["steps_to_go"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)
                act_pri_index = batch["act_pri_index"].to(self.device)  # (B, T)

                
                # VLM encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = VLM_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                task_inst_emb = VLM_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)
                
                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)

                # act_pri sees only the mid window (no start frame, no rewind); see _slice_act_pri_inputs
                act_pri_img_emb, act_pri_state, act_pri_lens = self._slice_act_pri_inputs(img_emb, state)
                act_pri_pred, class_pred = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)
                act_pri_pred = F.softmax(act_pri_pred, dim=-1)
                act_pri_pred_conf, act_pri_pred_idx = torch.max(act_pri_pred, dim=-1)  # (B,)
                lang_emb = self.act_pri_feature[act_pri_pred_idx].to(self.device)
                gate_idx = self.task_to_class_id[act_pri_pred_idx]  # (B,)


                reward_pred, aux_loss, info = reward_model(img_emb, lang_emb, task_inst_emb, state, lens, gate_idx=gate_idx)
                    
               
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                smoothed_item = raw_item
                
                if cfg.model.use_future_step:
                    gt_ep_result.append(-steps_to_go[0, cfg.model.n_obs_steps].item())
                    pred_ep_result.append(-raw_item)
                    pred_ep_smoothed.append(-smoothed_item)
                else:
                    # gt_ep_result.append(normalize_dense(trg[0, cfg.model.n_obs_steps].item()))
                    gt_ep_result.append(trg[0, cfg.model.n_obs_steps].item())
                    pred_ep_result.append(raw_item)
                    pred_ep_smoothed.append(smoothed_item)
                
                gt_act_pri_result.append(act_pri_index[0, -1].item()) 
                pred_act_pri_result.append(act_pri_pred_idx[0].item())
                pred_act_pri_conf.append(act_pri_pred_conf[0].item())
                aux_loss_result.append(aux_loss.item())
                expert_load_result.append(info["expert_load"].cpu().numpy())
                
                
            # save results
            task_name = lang_strs
            save_dir = plot_episode_result(ep_index, 
                                           pred_ep_smoothed, 
                                           gt_ep_result, 
                                           x_offset, 
                                           rollout_save_dir, 
                                           frame_gap=eval_frame_gap,
                                           expert_load_result=expert_load_result,
                                           task_name=task_name, 
                                           top_k=cfg.model.moe.top_k,
                                           split_task=split_task
                                           )
            act_pri_loss_ep = plot_act_pri_result(ep_index,
                                pred_act_pri_result,
                                gt_act_pri_result,
                                self.task_to_class_id,
                                self.task_list,
                                self.class_list,
                                x_offset,
                                rollout_save_dir,
                                frame_gap=eval_frame_gap,
                                task_name=task_name,
                                split_task=split_task,
                                ep_conf=pred_act_pri_conf
                                )
            np.save(Path(save_dir) / "pred_act_pri.npy", np.array(pred_act_pri_result))
            np.save(Path(save_dir) / "pred_act_pri_conf.npy", np.array(pred_act_pri_conf))
            np.save(Path(save_dir) / "gt_act_pri.npy", np.array(gt_act_pri_result))
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))
            np.save(Path(save_dir) / "gt.npy", np.array(gt_ep_result))
            np.save(Path(save_dir) / "smoothed.npy", np.array(pred_ep_smoothed))
            np.save(Path(save_dir) / "aux_loss.npy", np.array(aux_loss_result))
            np.save(Path(save_dir) / "expert_load.npy", np.array(expert_load_result))
            print(f"[Eval Video] episode_{ep_index} making video...")
            chunk_id = ep_index // 1000
            root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
            middle_video_dir = root / f"videos/chunk-{chunk_id:03d}/top_camera-images-rgb"
            if cfg.eval.eval_whole_task_from_act_pri:
                generate_whole_task_reward(save_dir, act_pri_th=cfg.eval.act_pri_th)
            
            
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap,
                              task_name=task_name,
                              split_task=split_task,
                              eval_whole_task_from_act_pri=cfg.eval.eval_whole_task_from_act_pri,
                              )
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}, progress: {count+1} / {cfg.eval.run_times}")

            
            return act_pri_loss_ep


    def eval_raw_data(self):
        import random
        cfg = self.cfg
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # SigLIP encoder
        VLM_encoder = FrozenSiglipEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_dim = 768; txt_dim = 768
        
        self.act_pri_feature = VLM_encoder.encode_text(self.task_list).to(self.device)  # (num_tasks, txt_dim)

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model,
                                  vis_emb_dim=vis_dim,
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  num_cameras=len(self.camera_names),
                                   # === Multi-gate MoE ===
                                  num_gates=cfg.model.num_class,
                                   # === MoE hyper-parameters ===
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
                                  ).to(self.device)

        act_pri_model = ActionTransformer(d_model=cfg.model.act_pri.d_model, 
                                    vis_emb_dim=vis_dim, 
                                    state_dim=cfg.model.act_pri.state_dim,
                                    n_layers=cfg.model.act_pri.n_layers,
                                    n_heads=cfg.model.act_pri.n_heads,
                                    dropout=cfg.model.act_pri.dropout,
                                    num_cameras=len(self.camera_names),
                                    num_tasks=cfg.model.num_tasks,
                                    num_classes=cfg.model.num_class,
                                  ).to(self.device)

        # Load checkpoints
        reward_model_path = Path(cfg.eval.ckpt_path_reward)
        act_pri_model_path = Path(cfg.eval.ckpt_path_act_pri)
        
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        reward_model.load_state_dict(reward_ckpt["model"])
        reward_model.to(self.device)
        reward_model.eval()

        act_pri_ckpt = torch.load(act_pri_model_path, map_location=self.device)
        act_pri_model.load_state_dict(act_pri_ckpt["model"])
        act_pri_model.to(self.device)
        act_pri_model.eval()

        # save path
        rollout_save_dir = Path(self.save_dir) / "eval_video"
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")

        
        x_offset = 0
        # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
        data_dir = cfg.eval.raw_data_dir
        run_times = cfg.eval.raw_data_run_times
        # Get all valid episode paths
        all_episodes = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("episode_")
        ]
        eval_list = all_episodes
        
        random.seed(cfg.general.seed)
        # randomly select eval_list
        if len(all_episodes) >= run_times:
            eval_list = random.sample(all_episodes, run_times)
        else:
            raise ValueError(f"Not enough episodes in {data_dir} to sample {run_times} items.")


        for i in range(run_times):
            data_path = eval_list[i]
            pred_ep_result = []
            pred_ep_smoothed = []
            # randomly select 
            ep_index = os.path.basename(data_path)
            frame_num = get_frame_num(data_path)
            traj_joint_data = get_traj_data(data_path)
            eval_frame_gap = cfg.eval.eval_frame_gap
            print(f"[EVAL_RAW]: process {i+1}/{run_times} episode: {ep_index}")
            for idx in tqdm(range(0, frame_num, eval_frame_gap), desc=f"Processing data"):
                batch = get_frame_data_fast(path=data_path, 
                                    traj_joint_data=traj_joint_data, 
                                    idx=idx,
                                    n_obs_steps=cfg.model.n_obs_steps,
                                    frame_gap=cfg.model.frame_gap,
                                    max_rewind_steps=cfg.model.max_rewind_steps,
                                    camera_names=cfg.general.camera_names,
                                    device=self.device)
                
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = ["fold the tshirt"]
                lens = torch.tensor([1+cfg.model.n_obs_steps], dtype=torch.int32, device=self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)

                # VLM encoding
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = VLM_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                task_inst_emb = VLM_encoder.encode_text(lang_strs)  # (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)

                # act_pri sees only the mid window (no start frame, no rewind); see _slice_act_pri_inputs
                act_pri_img_emb, act_pri_state, act_pri_lens = self._slice_act_pri_inputs(img_emb, state)
                act_pri_pred, class_pred = act_pri_model(act_pri_img_emb, act_pri_state, act_pri_lens)
                act_pri_pred_idx = torch.argmax(act_pri_pred, dim=-1)  # (B,)
                lang_emb = self.act_pri_feature[act_pri_pred_idx].to(self.device)
                gate_idx = self.task_to_class_id[act_pri_pred_idx]  # (B,)

                reward_pred, _, _ = reward_model(img_emb, lang_emb, task_inst_emb, state, lens, gate_idx=gate_idx)  # (B, T)
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                smoothed_item = raw_item
                
                pred_ep_result.append(raw_item)
                pred_ep_smoothed.append(smoothed_item)
                

            # save results
            save_dir = plot_episode_result_raw_data(ep_index, pred_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_smoothed=None)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))

            print(f"[Eval Video] episode_{ep_index} making video...")
            middle_video_dir = Path(f"{data_path}/top_camera-images-rgb.mp4")
            try:
                produce_video(save_dir=rollout_save_dir, 
                              middle_video=middle_video_dir, 
                              episode_num=ep_index, 
                              raw_data=True,
                              x_offset=x_offset, 
                              frame_gap=eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}")


    