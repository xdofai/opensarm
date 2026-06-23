import os
import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.normalizer import SingleFieldLinearNormalizer
import json
import numpy as np

def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_ckpt(model, opt, ep, save_dir, input_name=None):
    save_dir = Path(save_dir) / "checkpoints"  # convert to Path first
    name = f"{input_name}.pt" if input_name else f"epoch{ep:04d}.pt"
    p = save_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(model=model.state_dict(),
             optimizer=opt.state_dict(),
             epoch=ep),
        p
    )

@torch.no_grad()
def get_normalizer_from_calculated(path, device) -> "SingleFieldLinearNormalizer":
    """
    Load norm stats from a JSON file. Accepts absolute or relative paths.
    Relative paths are resolved robustly even when Hydra changes the CWD.
    """
    # --- Resolve path robustly ---
    original = str(path)
    abs_path = None

    # 1) Try Hydra's original working dir resolution (best for Hydra runs)
    try:
        from hydra.utils import to_absolute_path
        candidate = Path(to_absolute_path(original))
        if candidate.exists():
            abs_path = candidate
    except Exception:
        pass

    # 2) If already absolute and exists, use it
    if abs_path is None:
        candidate = Path(original)
        if candidate.is_absolute() and candidate.exists():
            abs_path = candidate

    # 3) Try common relative bases: current CWD, this file's dir, and a couple of parents
    if abs_path is None and not Path(original).is_absolute():
        bases = []
        try:
            here = Path(__file__).resolve().parent
            bases.extend([Path.cwd(), here, here.parent, here.parent.parent])
        except Exception:
            bases.append(Path.cwd())

        for base in bases:
            candidate = base / original
            if candidate.exists():
                abs_path = candidate
                break

    if abs_path is None:
        tried = [f"- Hydra to_absolute_path('{original}')",
                 f"- Absolute '{original}'" if Path(original).is_absolute() else f"- '{Path.cwd() / original}' (CWD)",
                 "- __file__/.. variations"]
        raise FileNotFoundError(
            f"Could not locate normalizer JSON '{original}'. Tried:\n" + "\n".join(tried)
        )

    # --- Load and build normalizer ---
    with open(abs_path, "r") as f:
        norm_data = json.load(f)["norm_stats"]

    def to_tensor_slice(data, k: int = 14):  # both arms
        return torch.tensor(data[:k], dtype=torch.float32, device=device)

    state_stats = norm_data["state"]
    state_std = to_tensor_slice(state_stats["std"])
    state_mean = to_tensor_slice(state_stats["mean"])

    state_normalizer = SingleFieldLinearNormalizer.create_manual(
        scale=1.0 / state_std,
        offset=-(state_mean / state_std),
        input_stats_dict={
            "min": to_tensor_slice(state_stats["q01"]),
            "max": to_tensor_slice(state_stats["q99"]),
            "mean": state_mean,
            "std": state_std,
        },
    )
    return state_normalizer


def plot_episode_result(ep_index,
                        ep_result,
                        gt_ep_result,
                        x_offset,
                        rollout_save_dir,
                        frame_gap=None, ep_conf=None, ep_smoothed=None,
                        expert_load_result=None,
                        gate_load_result=None,
                        task_name=None,
                        top_k=2,
                        split_task=False):

    if split_task and task_name:
        save_dir = rollout_save_dir / f"{task_name}" / f"episode_{ep_index}"
    else:
        save_dir = rollout_save_dir / f"episode_{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Trim initial frames
    ep_result = ep_result[x_offset:]
    gt_ep_result = gt_ep_result[x_offset:]

    # Convert to numpy arrays
    ep_result_np = np.array(ep_result)
    gt_ep_result_np = np.array(gt_ep_result)
    ep_conf_np = np.asarray(ep_conf)[x_offset:] if ep_conf is not None else None
    ep_smoothed_np = np.asarray(ep_smoothed)[x_offset:] if ep_smoothed is not None else None
    if ep_smoothed_np is not None:
        ep_result_np = ep_smoothed_np

    # === Timesteps ===
    if frame_gap is None:
        timesteps = np.arange(len(ep_result_np)) + x_offset
    else:
        timesteps = np.arange(0, len(ep_result_np) * frame_gap, frame_gap) + x_offset * frame_gap

    # Compute MSE and MAE
    mse = np.mean((ep_result_np - gt_ep_result_np) ** 2)
    mae = np.mean(np.abs(ep_result_np - gt_ep_result_np))

    # Plot
    plt.figure()
    plt.plot(timesteps, ep_result_np, label="Predicted")
    plt.plot(timesteps, gt_ep_result_np, label="GT")
    if ep_conf_np is not None:
        plt.plot(timesteps, ep_conf_np, label="Conf", linestyle=":", color="green")
    # Add dummy lines for metrics in the legend
    plt.plot([], [], ' ', label=f"MSE: {mse:.4f}")
    plt.plot([], [], ' ', label=f"MAE: {mae:.4f}")
    plt.title("Episode Result")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "plot.png")

    if expert_load_result is not None:
        loads = np.asarray(expert_load_result)
        plt.figure()
        for i in range(loads.shape[1]):
            plt.plot(timesteps, loads[:, i], label=f"Expert {i}")
        plt.title("Expert Load Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Expert weights")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "expert_load_plot.png")

        if gate_load_result is not None:
            gate_loads = np.asarray(gate_load_result)
            plt.figure()
            for i in range(gate_loads.shape[1]):
                plt.plot(timesteps, gate_loads[:, i], label=f"Gate {i}")
            plt.title("Gate Load Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Gate weights")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / "gate_load_plot.png")

        if task_name is not None:
            avg_weights = loads.mean(axis=0)               # (8,)
            topk_idx = np.argsort(-avg_weights)[:top_k]

            # --- Write summary file ---
            summary_path = save_dir / "episode_summary.txt"
            with open(summary_path, "w") as f:
                f.write(f"task_name: {task_name}\n")
                f.write(f"mse: {mse:.6f}\n")
                f.write(f"mae: {mae:.6f}\n")
                f.write("expert_avg_weights:\n")
                for i, w in enumerate(avg_weights):
                    f.write(f"  expert_{i}: {w:.4f}\n")
                f.write(f"top{top_k}_experts:\n")
                for rank, i in enumerate(topk_idx, start=1):
                    f.write(f"  #{rank}: expert_{i} (avg_weight={avg_weights[i]:.4f})\n")

    plt.close()

    return str(save_dir)


def plot_episode_result_raw_data(ep_index, ep_result, x_offset, rollout_save_dir, frame_gap=None, ep_conf=None, ep_smoothed=None, task_name=None, split_task=False):
    if split_task and task_name:
        save_dir = rollout_save_dir / f"{task_name}" / f"{ep_index}"
    else:
        save_dir = rollout_save_dir / f"{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Trim & to numpy ===
    ep_result_np = np.asarray(ep_result)[x_offset:]  # raw predictions
    ep_conf_np = np.asarray(ep_conf)[x_offset:] if ep_conf is not None else None
    ep_smoothed_np = np.asarray(ep_smoothed)[x_offset:] if ep_smoothed is not None else None

    # === Handle empty results ===
    if len(ep_result_np) == 0:
        print(f"Warning: Episode {ep_index} has no results after trimming. Skipping plot.")
        return None

    # === Timesteps ===
    if frame_gap is None:
        timesteps = np.arange(len(ep_result_np)) + x_offset
    else:
        timesteps = np.arange(0, len(ep_result_np) * frame_gap, frame_gap) + x_offset * frame_gap

    # === Plot ===
    fig, ax = plt.subplots(figsize=(4.48, 4.48), dpi=100)  # 448x448 px
    line_pred, = ax.plot(timesteps, ep_result_np, label="Raw Predicted", linewidth=2)
    handles, labels = [line_pred], ["Raw Predicted"]

    if ep_smoothed_np is not None:
        line_smooth, = ax.plot(timesteps, ep_smoothed_np, label="Smoothed", linewidth=2, color="orange")
        handles.append(line_smooth)
        labels.append("Smoothed")

    ax.set_title("Episode Result")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Prediction")
    ax.grid(True)

    # Confidence on twin y-axis
    if ep_conf_np is not None:
        ax2 = ax.twinx()
        line_conf, = ax2.plot(timesteps, ep_conf_np, linestyle=":", label="Confidence", color="green")
        ax2.set_ylabel("Confidence")
        # Merge legends
        handles.append(line_conf)
        labels.append("Confidence")

    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    out_path = save_dir / "plot.png"
    fig.savefig(out_path)
    plt.close(fig)

    return str(save_dir)

def plot_act_pri_result(ep_index,
                        act_pri_result,
                        gt_act_pri_result,
                        task_to_class_id,
                        task_list,
                        class_list,
                        x_offset,
                        rollout_save_dir,
                        frame_gap=None,
                        task_name=None,
                        split_task=False,
                        ep_conf=None,
                        raw=False,
                        seg_boundaries=None):
    
    # 1. Setup Save Directory
    if split_task and task_name:
        if raw:
            save_dir = rollout_save_dir / f"{task_name}" / f"{ep_index}"
        else:
            save_dir = rollout_save_dir / f"{task_name}" / f"episode_{ep_index}"
    else:
        if raw:
            save_dir = rollout_save_dir / f"{ep_index}"
        else:
            save_dir = rollout_save_dir / f"episode_{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Reorder task_list by Class for Grouping
    mapping = task_to_class_id.cpu().numpy()
    
    # Create a list of (task_index, class_index)
    task_class_pairs = [(i, mapping[i]) for i in range(len(task_list))]
    # Sort by class_index first, then original index
    sorted_pairs = sorted(task_class_pairs, key=lambda x: (x[1], x[0]))
    
    # New ordered indices and labels for the Y-axis
    new_order_indices = [pair[0] for pair in sorted_pairs]
    sorted_task_labels = [task_list[i] for i in new_order_indices]
    
    # Create a lookup: original_index -> new_y_coordinate
    index_to_y = {orig_idx: new_y for new_y, orig_idx in enumerate(new_order_indices)}

    # 3. Data Preparation
    pred_np = np.array(act_pri_result)[x_offset:]
    gt_np = np.array(gt_act_pri_result)[x_offset:]
    
    # Map predictions and GT to their new Y positions
    pred_y = np.array([index_to_y[int(p)] for p in pred_np])
    gt_y = np.array([index_to_y[int(g)] for g in gt_np])
    
    if frame_gap is None:
        timesteps = np.arange(len(pred_np)) + x_offset
    else:
        timesteps = np.arange(0, len(pred_np) * frame_gap, frame_gap) + x_offset * frame_gap

    # 4. Metric Calculation
    losses = [0.0 if p == g else (0.5 if mapping[int(p)] == mapping[int(g)] else 1.0) 
              for p, g in zip(pred_np, gt_np)]
    mean_h_loss = np.mean(losses)
    accuracy = np.mean(pred_np == gt_np) * 100

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Horizontal Background Colors for each Class
    cmap = plt.get_cmap('Pastel2', len(class_list))
    
    # Find start and end Y-indices for each class to draw axhspan
    for c_idx in range(len(class_list)):
        class_y_coords = [y for y, pair in enumerate(sorted_pairs) if pair[1] == c_idx]
        if class_y_coords:
            y_min, y_max = min(class_y_coords), max(class_y_coords)
            # Add 0.5 padding to cover the whole text area
            ax.axhspan(y_min - 0.5, y_max + 0.5, facecolor=cmap(c_idx), alpha=0.3, 
                       label=f"Class {c_idx}")

    # Plot lines using the new Y coordinates
    ax.plot(timesteps, gt_y, label="Ground Truth", color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(timesteps, pred_y, label="Prediction", color='blue', marker='o', markersize=3, linewidth=1, alpha=0.7)

    # 6. Formatting
    ax.set_yticks(range(len(sorted_task_labels)))
    ax.set_yticklabels(sorted_task_labels, fontsize=9)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Action Primitive (Grouped by Class)", fontsize=12)
    
    title_str = f"Action: {task_name} | Episode {ep_index} | Mean Loss: {mean_h_loss:.3f} | Accuracy: {accuracy:.1f}%"
    ax.set_title(title_str, fontsize=14, fontweight='bold')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    if seg_boundaries:
        for boundary in seg_boundaries:
            ax.axvline(x=boundary, color='green', linestyle='--', linewidth=2.0, alpha=0.6)

    ax.grid(axis='both', linestyle=':', alpha=0.4)
    plt.tight_layout()

    plt.savefig(save_dir / "act_pri_loss.png", dpi=200)
    plt.close()
    
    if ep_conf is not None:
        # Ensure ep_conf is sliced same as pred_np
        ep_conf_np = np.asarray(ep_conf)[x_offset:]
        save_conf_plot(timesteps, ep_conf_np, save_dir, task_name, ep_index)
    
    return mean_h_loss



def save_conf_plot(timesteps, act_pri_conf, save_dir, task_name, ep_index):
    """
    Plots the confidence values over time and saves to act_pri_conf.png.
    """
    # 1. Data Preparation
    conf_np = np.asarray(act_pri_conf)
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the confidence curve
    ax.plot(timesteps, conf_np, color='darkorange', linewidth=1.5, label='Model Confidence')
    
    # Add a fill-under effect for better visualization
    ax.fill_between(timesteps, conf_np, 0, color='orange', alpha=0.1)
    
    # Add a reference line for common threshold (e.g., 0.5)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Threshold (0.5)')
    
    # 3. Formatting
    ax.set_ylim(0, 1.05) # Confidence is always 0-1
    ax.set_xlim(timesteps[0], timesteps[-1])
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Confidence Score", fontsize=10)
    ax.set_title(f"Confidence Profile: {task_name} (Ep {ep_index})", fontsize=12, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    # 4. Save
    conf_save_path = save_dir / "act_pri_conf.png"
    plt.savefig(conf_save_path, dpi=200)
    plt.close()

def generate_whole_task_reward(path, act_pri_th=3):
    """
    Detects task transitions based on pred_act_pri and accumulates rewards/progress.
    
    Args:
        path (str): Directory containing pred_act_pri.npy and pred.npy.
        act_pri_th (int): Threshold for consecutive IDs to be considered a new action.
    """
    # 1. Load data
    act_pri_file = os.path.join(path, "pred_act_pri.npy")
    pred_file = os.path.join(path, "pred.npy")
    
    if not os.path.exists(act_pri_file) or not os.path.exists(pred_file):
        print(f"Error: Files not found in {path}")
        return

    pred_act_pri = np.load(act_pri_file)
    pred = np.load(pred_file)

    # 2. Align sizes
    min_size = min(len(pred_act_pri), len(pred))
    pred_act_pri = pred_act_pri[:min_size]
    pred = pred[:min_size]
    
    # Initialize new_progress with original pred values
    new_progress = pred.copy().astype(np.float64)
    
    # 3. Detect transitions and accumulate
    # We look for a point i where the next (act_pri_th + 1) elements 
    # are all different from the element at i-1.
    i = 1
    prev_id = pred_act_pri[0]
    while i <= min_size - (act_pri_th + 1):
        # prev_id = pred_act_pri[i-1]
        # Check if the next 'act_pri_th + 1' elements are different from prev_id
        window = pred_act_pri[i : i + act_pri_th + 1]
        
        if np.all(window != prev_id) and np.all(window == window[0]):
            prev_id = pred_act_pri[i]
            # Transition detected at index i
            # new_progress[i:] += pred[i-1]
            new_progress[i:] += pred[i-1]
            
            # Skip the window to avoid multiple detections for the same transition
            i += (act_pri_th + 1)
        else:
            i += 1

    # 4. Save the processed data
    output_npy_path = os.path.join(path, "accumulated_pred.npy")
    np.save(output_npy_path, new_progress)

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    # plt.plot(pred, label='Original Pred', alpha=0.6, linestyle='--')
    plt.plot(new_progress, label='Accumulated Progress', linewidth=2)
    plt.title("Accumulated Task Progress Visualization")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    output_png_path = os.path.join(path, "accumulated_progress.png")
    plt.savefig(output_png_path)
    plt.close()
