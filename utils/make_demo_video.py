import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
from typing import List, Optional, Union

def produce_video(
    save_dir: Union[str, Path],
    middle_video: Union[str, Path],
    episode_num: Optional[int] = None,
    *,
    raw_data: bool = False,
    x_offset: int = 30,
    frame_gap: Optional[int] = None,
    frame_rate: int = 30,
    target_h: int = 448,
    target_w: int = 448,
):
    """
    Create a side-by-side video panel with the middle camera view and a progress plot.

    Two modes (controlled by `raw_data`):
      - raw_data=False (episode mode): expects per-episode folder structure:
            save_dir/episode_{episode_num}/pred.npy
            save_dir/episode_{episode_num}/conf.npy (optional)
            save_dir/episode_{episode_num}/smoothed.npy (optional)
            save_dir/episode_{episode_num}/gt.npy (optional)
        and a middle video at:
            middle_video_dir/episode_{episode_num:06d}.mp4
        where `middle_video` is the directory containing the per-episode .mp4 files.

      - raw_data=True (raw-data mode): expects files directly in `save_dir`:
            save_dir/pred.npy
            save_dir/conf.npy (optional)
            save_dir/smoothed.npy (optional)
        and `middle_video` is the full path to the video file.

    Args:
        save_dir: Root directory for saving / reading arrays.
        middle_video: Either a directory (episode mode) or a full video path (raw-data mode).
        episode_num: Episode index (required if raw_data=False).
        raw_data: Switch between episode mode (False) and raw-data mode (True).
        x_offset: Number of initial prediction steps to skip.
        frame_gap: If provided, skip frames from the video by this factor when applying x_offset.
        frame_rate: FPS used when reading frames and writing output.
        target_h, target_w: Per-panel resolution.

    Output:
        Writes a combined video to:
          - episode mode: save_dir/episode_{episode_num}/combined_video.mp4
          - raw-data mode: save_dir/combined_video.mp4
    """
    save_dir = Path(save_dir)

    # -------------------------
    # Resolve paths per mode
    # -------------------------
    if raw_data:
        episode_dir = save_dir / episode_num
        middle_video_path = Path(middle_video)
        output_path = episode_dir / "combined_video.mp4"
        
        pred_path   = episode_dir / "pred.npy"
        conf_path   = episode_dir / "conf.npy"
        smooth_path = episode_dir / "smoothed.npy"
        gt_path     = episode_dir / "gt.npy"  # optional; not required in raw mode
    else:
        if episode_num is None:
            raise ValueError("`episode_num` must be provided when raw_data=False.")
        episode_dir = save_dir / f"episode_{episode_num}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        middle_video_dir = Path(middle_video)
        middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
        output_path = episode_dir / "combined_video.mp4"

        pred_path   = episode_dir / "pred.npy"
        conf_path   = episode_dir / "conf.npy"
        smooth_path = episode_dir / "smoothed.npy"
        gt_path     = episode_dir / "gt.npy"  # optional

    # -------------------------
    # Load arrays
    # -------------------------
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    pred_full = np.load(pred_path)
    pred = pred_full[x_offset:]

    # confidence (optional)
    conf = None
    if conf_path.exists():
        conf = np.load(conf_path)[x_offset:]

    # smoothed (optional, fallback to pred)
    if smooth_path.exists():
        smoothed = np.load(smooth_path)[x_offset:]
    else:
        smoothed = pred

    # ground truth (optional; used only in episode mode originally, but we just load if present)
    gt = None
    if gt_path.exists():
        gt_full = np.load(gt_path)
        gt = gt_full[x_offset:]

    T = len(pred)

    # -------------------------
    # Load and align video frames
    # -------------------------
    if not middle_video_path.exists():
        raise FileNotFoundError(f"Missing video file: {middle_video_path}")

    clip_middle = VideoFileClip(str(middle_video_path))
    raw_frames = list(clip_middle.iter_frames(fps=frame_rate))

    # Apply x_offset at the frame level (optionally scaled by frame_gap)
    start_idx = x_offset * frame_gap if frame_gap is not None else x_offset
    if start_idx >= len(raw_frames):
        raise ValueError(
            f"x_offset ({x_offset}) with frame_gap ({frame_gap}) exceeds video length ({len(raw_frames)})."
        )
    frames_middle = raw_frames[start_idx:]

    # Align prediction length with available frames
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(
            f"WARNING: Not enough frames in video. Expected {T}, found {min_frames_num}. "
            f"Truncating predictions by {gap} to match video."
        )
        T = min_frames_num
        pred = pred[gap:]
        if conf is not None:
            conf = conf[gap:]
        if smoothed is not None:
            smoothed = smoothed[gap:]
        if gt is not None:
            gt = gt[gap:]

    # Uniformly sample/align frames to match T
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]

    # -------------------------
    # Compose panels
    # -------------------------
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame(
            t,
            pred,
            x_offset,
            height=target_h,
            width=target_w,
            frame_gap=frame_gap,
            smoothed=smoothed,
        )
        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # -------------------------
    # Write video
    # -------------------------
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec="libx264")


def draw_plot_frame(step: int, pred, x_offset, width=448, height=448, frame_gap=None, smoothed=None):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    if frame_gap is None:
        timesteps = np.arange(len(pred)) + x_offset
    else:
        timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap
        
    # === Plot raw prediction ===
    pred = smoothed if smoothed is not None else pred
    line_pred, = ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    handles, labels = [line_pred], ["Predicted"]

    # === Vertical line at current step ===
    if frame_gap is None:
        ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=step * frame_gap + x_offset * frame_gap, color='r', linestyle='--', linewidth=2)

    # === Labels and style ===
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # === Legend ===
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img




def produce_video_act_pri(
    save_dir: Union[str, Path],
    middle_video: Union[str, Path],
    task_to_class_id,
    task_list: List[str],
    class_list: List[str],
    episode_num: Optional[int] = None,
    task_name: Optional[str] = None,
    split_task: bool = False,
    *,
    raw_data: bool = False,
    x_offset: int = 30,
    frame_gap: Optional[int] = None,
    frame_rate: int = 30,
    target_h: int = 480,
    target_w: int = 640,
    seg_boundaries=None,
):
    """
    Create a side-by-side video panel with the middle camera view and an
    action-primitive plot styled like plot_act_pri_result().

    Expects:
      - pred_act_pri.npy
      - gt_act_pri.npy (optional; falls back to pred if missing)
    """
    save_dir = Path(save_dir)

    # -------------------------
    # Resolve paths per mode
    # -------------------------
    if raw_data:
        episode_dir = save_dir / episode_num
        if task_name and split_task:
            episode_dir = save_dir / task_name / episode_num
        episode_dir.mkdir(parents=True, exist_ok=True)
        middle_video_path = Path(middle_video)
        output_path = episode_dir / "combined_video_act_pri.mp4"
        pred_path = episode_dir / "pred_act_pri.npy"
        gt_path = episode_dir / "gt_act_pri.npy"
    else:
        if episode_num is None:
            raise ValueError("`episode_num` must be provided when raw_data=False.")
        if split_task and task_name:
            episode_dir = save_dir / f"{task_name}" / f"episode_{episode_num}"
        else:
            episode_dir = save_dir / f"episode_{episode_num}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        middle_video_dir = Path(middle_video)
        middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
        output_path = episode_dir / "combined_video_act_pri.mp4"

        pred_path = episode_dir / "pred_act_pri.npy"
        gt_path = episode_dir / "gt_act_pri.npy"

    # -------------------------
    # Load arrays
    # -------------------------
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    pred_full = np.load(pred_path)
    pred_np = np.asarray(pred_full)[x_offset:]

    if gt_path.exists():
        gt_full = np.load(gt_path)
        gt_np = np.asarray(gt_full)[x_offset:]
    else:
        gt_np = pred_np

    T = min(len(pred_np), len(gt_np))
    pred_np = pred_np[:T]
    gt_np = gt_np[:T]

    # -------------------------
    # Precompute plot context
    # -------------------------
    mapping = task_to_class_id.cpu().numpy() if hasattr(task_to_class_id, "cpu") else np.asarray(task_to_class_id)
    task_class_pairs = [(i, mapping[i]) for i in range(len(task_list))]
    sorted_pairs = sorted(task_class_pairs, key=lambda x: (x[1], x[0]))
    new_order_indices = [pair[0] for pair in sorted_pairs]
    sorted_task_labels = [task_list[i] for i in new_order_indices]
    index_to_y = {orig_idx: new_y for new_y, orig_idx in enumerate(new_order_indices)}

    losses = [
        0.0 if p == g else (0.5 if mapping[int(p)] == mapping[int(g)] else 1.0)
        for p, g in zip(pred_np, gt_np)
    ]
    mean_h_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    accuracy = float(np.mean(pred_np == gt_np) * 100) if len(pred_np) > 0 else 0.0
    title_str = (
        f"Action: {task_name} | Episode {episode_num} | Mean Loss: {mean_h_loss:.3f} | Accuracy: {accuracy:.1f}%"
    )

    plot_ctx = {
        "mapping": mapping,
        "sorted_pairs": sorted_pairs,
        "sorted_task_labels": sorted_task_labels,
        "index_to_y": index_to_y,
        "class_list": class_list,
        "title_str": title_str,
        "seg_boundaries": seg_boundaries or [],
    }

    # -------------------------
    # Load and align video frames
    # -------------------------
    if not middle_video_path.exists():
        raise FileNotFoundError(f"Missing video file: {middle_video_path}")

    clip_middle = VideoFileClip(str(middle_video_path))
    raw_frames = list(clip_middle.iter_frames(fps=frame_rate))

    start_idx = x_offset * frame_gap if frame_gap is not None else x_offset
    if start_idx >= len(raw_frames):
        raise ValueError(
            f"x_offset ({x_offset}) with frame_gap ({frame_gap}) exceeds video length ({len(raw_frames)})."
        )
    frames_middle = raw_frames[start_idx:]

    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(
            f"WARNING: Not enough frames in video. Expected {T}, found {min_frames_num}. "
            f"Truncating predictions by {gap} to match video."
        )
        T = min_frames_num
        pred_np = pred_np[gap:]
        gt_np = gt_np[gap:]

    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]

    # -------------------------
    # Compose panels
    # -------------------------
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_act_pri_plot_frame(
            t,
            pred_np,
            gt_np,
            x_offset,
            plot_ctx,
            height=target_h,
            width=target_w,
            frame_gap=frame_gap,
        )
        combined = np.concatenate((middle_resized, plot_img), axis=0)
        combined_frames.append(combined)

    # -------------------------
    # Write video
    # -------------------------
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec="libx264")


def draw_act_pri_plot_frame(
    step: int,
    pred_np,
    gt_np,
    x_offset,
    plot_ctx,
    width=448,
    height=448,
    frame_gap=None,
):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    if frame_gap is None:
        timesteps = np.arange(len(pred_np)) + x_offset
    else:
        timesteps = np.arange(0, len(pred_np) * frame_gap, frame_gap) + x_offset * frame_gap

    pred_y = np.array([plot_ctx["index_to_y"][int(p)] for p in pred_np])
    gt_y = np.array([plot_ctx["index_to_y"][int(g)] for g in gt_np])

    cmap = plt.get_cmap("Pastel2", len(plot_ctx["class_list"]))
    for c_idx in range(len(plot_ctx["class_list"])):
        class_y_coords = [y for y, pair in enumerate(plot_ctx["sorted_pairs"]) if pair[1] == c_idx]
        if class_y_coords:
            y_min, y_max = min(class_y_coords), max(class_y_coords)
            ax.axhspan(y_min - 0.5, y_max + 0.5, facecolor=cmap(c_idx), alpha=0.3, label=f"Class {c_idx}")

    ax.plot(timesteps, gt_y, label="Ground Truth", color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(timesteps, pred_y, label="Prediction", color='blue', marker='o', markersize=3, linewidth=1, alpha=0.7)

    for boundary in plot_ctx.get("seg_boundaries", []):
        ax.axvline(x=boundary, color='green', linestyle='--', linewidth=2.0, alpha=0.6)

    if frame_gap is None:
        ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=step * frame_gap + x_offset * frame_gap, color='r', linestyle='--', linewidth=2)

    ax.set_yticks(range(len(plot_ctx["sorted_task_labels"])))
    ax.set_yticklabels(plot_ctx["sorted_task_labels"], fontsize=9)
    ax.set_xlabel("Time Step", fontsize=12)
    # ax.set_ylabel("Action Primitive (Grouped by Class)", fontsize=12)
    ax.set_title(plot_ctx["title_str"], fontsize=14, fontweight='bold')

    ax.grid(axis='both', linestyle=':', alpha=0.4)
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)

    return img
