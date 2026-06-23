"""Generate validation videos overlaying the progress curve on the top camera video.

Randomly selects episodes that have progress.npy, renders the top camera video
with a progress plot overlay, and saves to DATA_DIR/validate_video/.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import json

import numpy as np

TOP_MP4 = "top_camera-images-rgb.mp4"


def load_episode_dirs_from_results(results_path: Path) -> list[Path]:
    """Extract episode directories from a results JSONL, keeping last label per episode."""
    episodes = {}
    with results_path.open() as f:
        for line in f:
            row = json.loads(line)
            ep_dir = Path(row["video_path"]).parent
            episodes[str(ep_dir)] = ep_dir
    # Only keep episodes that have progress.npy
    return [d for d in episodes.values() if (d / "progress.npy").exists()]


def render_progress_frames(progress: np.ndarray, kept_indices: list[int], width: int, height: int) -> list[np.ndarray]:
    """Render progress plot as RGBA frames for the given frame indices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames = []
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    n_total = len(progress)
    t = np.arange(n_total) / max(n_total - 1, 1)

    for i in kept_indices:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time", fontsize=9)
        ax.set_ylabel("Progress", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_facecolor((0, 0, 0, 0.3))
        fig.patch.set_alpha(0)

        # Full curve in gray
        ax.plot(t, progress, color="#aaaaaa", linewidth=1.5)
        # Highlight up to current frame
        ax.plot(t[:i + 1], progress[:i + 1], color="#00ff88", linewidth=2.5)
        # Current point
        ax.plot(t[i], progress[i], "o", color="#ff4444", markersize=6)
        # Value text
        ax.text(0.98, 0.95, f"{progress[i]:.2f}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="white",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(int(fig.get_figheight() * 100), int(fig.get_figwidth() * 100), 4)
        frames.append(buf)

    plt.close(fig)
    return frames


def create_validation_video(episode_dir: Path, output_path: Path, speed: int = 10):
    """Create a side-by-side video: top camera + progress plot, sped up by skipping frames."""
    import cv2

    video_path = episode_dir / TOP_MP4
    progress = np.load(episode_dir / "progress.npy")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Align progress length to frame count
    if len(progress) != n_frames:
        progress = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(progress)),
            progress,
        )

    # Which frames to keep (every Nth frame)
    kept_indices = list(range(0, n_frames, speed))
    n_kept = len(kept_indices)

    # Plot dimensions: same height as video, width = 50% of video width
    plot_w = max(int(vid_w * 0.5), 200)
    plot_h = vid_h

    print(f"  Rendering {n_kept} frames ({n_frames} total, {speed}x speed)...")
    plot_frames = render_progress_frames(progress, kept_indices, plot_w, plot_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    plot_h_actual, plot_w_actual = plot_frames[0].shape[:2]

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (vid_w + plot_w_actual, max(vid_h, plot_h_actual)))

    frame_idx = 0
    kept_set = set(kept_indices)
    plot_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in kept_set:
            plot_rgba = plot_frames[plot_i]
            plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
            if plot_bgr.shape[0] != vid_h:
                plot_bgr = cv2.resize(plot_bgr, (plot_bgr.shape[1], vid_h))
            composite = np.hstack([frame, plot_bgr])
            writer.write(composite)
            plot_i += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode with ffmpeg for better compression
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], capture_output=True)

    Path(tmp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate validation videos with progress overlay")
    parser.add_argument("--results", required=True, help="Path to results JSONL file")
    parser.add_argument("-n", type=int, default=10, help="Number of episodes to sample (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    episodes = load_episode_dirs_from_results(results_path)
    if not episodes:
        raise SystemExit(f"No episodes with progress.npy found in {results_path}")

    print(f"Found {len(episodes)} episodes with progress.npy")

    n = min(args.n, len(episodes))
    rng = random.Random(args.seed)
    selected = rng.sample(episodes, n)

    validate_dir = results_path.parent / "validate_video"
    validate_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving validation videos to {validate_dir}\n")

    for i, ep in enumerate(selected):
        output_path = validate_dir / f"{ep.name}.mp4"
        print(f"[{i + 1}/{n}] {ep.name}")
        create_validation_video(ep, output_path)
        print(f"  -> {output_path}\n")

    print(f"Done. {n} validation videos saved to {validate_dir}")


if __name__ == "__main__":
    main()
