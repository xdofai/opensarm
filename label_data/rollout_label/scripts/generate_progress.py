"""Generate progress.npy and reward.npy for each episode based on labels.

Slope rules (relative to a base slope s):
  - progress_fast:  +2s
  - progress_slow:  +1s
  - adjust:          0
  - mistake:        -1s

Base slope s is solved so that integrating the piecewise-linear curve
from t=0 to t=T yields exactly `final_progress`.

If catastrophic failure is marked, progress after that frame is set to 0.
reward.npy = np.diff(progress, prepend=0) (per-step delta).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLOPE_MULT = {
    "progress_fast": 2.0,
    "progress_slow": 1.0,
    "adjust": 0.0,
    "mistake": -1.0,
    "catastrophic": 0.0,
}


def load_labels(results_path: Path) -> list[dict]:
    """Load all labels from a JSONL results file. Keep last label per episode."""
    labels = {}
    with results_path.open() as f:
        for line in f:
            row = json.loads(line)
            # Later entries override earlier (re-labels)
            labels[row["video_path"]] = row
    return list(labels.values())


def video_path_to_episode_dir(video_path: str) -> Path:
    """Extract episode directory from video path."""
    return Path(video_path).parent


def compute_progress(
    timestamp: np.ndarray,
    segments: list[dict],
    final_progress: float,
    catastrophic_frame: float | None,
    start_progress: float = 0.0,
) -> np.ndarray:
    """Compute progress curve aligned to timestamp array.

    Curve starts at ``start_progress`` and reaches ``final_progress`` at the end
    (or at ``catastrophic_frame`` if set, after which progress is zeroed).
    """
    n = len(timestamp)
    t = timestamp - timestamp[0]  # relative time from start

    # Map each timestep to its segment's slope multiplier
    multipliers = np.zeros(n, dtype=np.float64)
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        label = seg["label"]
        m = SLOPE_MULT.get(label, 0.0)
        mask = (t >= seg_start) & (t < seg_end)
        multipliers[mask] = m
    # Include the very last frame if it falls on a segment boundary
    if segments:
        last_label = segments[-1]["label"]
        multipliers[t >= segments[-1]["start"]] = SLOPE_MULT.get(last_label, 0.0)

    # Compute dt between consecutive timesteps
    dt = np.diff(t, prepend=0)
    dt[0] = 0  # first step has no delta

    if catastrophic_frame is not None:
        # Only count weighted sum up to catastrophic frame
        cat_mask = t < catastrophic_frame
        weighted_sum = np.sum(multipliers[cat_mask] * dt[cat_mask])
    else:
        weighted_sum = np.sum(multipliers * dt)

    delta = final_progress - start_progress

    if abs(weighted_sum) < 1e-9:
        if abs(delta) < 1e-9:
            progress = np.full(n, start_progress, dtype=np.float64)
        else:
            # No nonzero slopes but start != final — linear interpolation
            progress = np.linspace(start_progress, final_progress, n)
    else:
        base_slope = delta / weighted_sum
        # Integrate: progress[i] = start + sum of base_slope * m[j] * dt[j] for j<=i
        increments = base_slope * multipliers * dt
        progress = start_progress + np.cumsum(increments)

    # Clamp to [0, 1] for non-catastrophic
    if catastrophic_frame is None:
        progress = np.clip(progress, 0.0, 1.0)
    else:
        # Zero out after catastrophic frame
        cat_mask = t >= catastrophic_frame
        progress[cat_mask] = 0.0
        # Clip pre-catastrophic portion
        progress = np.clip(progress, 0.0, 1.0)

    return progress


def main():
    parser = argparse.ArgumentParser(description="Generate progress.npy and reward.npy from labels")
    parser.add_argument("--results", required=True, help="Path to results JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without saving")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    labels = load_labels(results_path)
    print(f"Loaded {len(labels)} labeled episodes from {results_path}")

    for row in labels:
        episode_dir = video_path_to_episode_dir(row["video_path"])
        ts_path = episode_dir / "timestamp.npy"

        if not ts_path.exists():
            print(f"[SKIP] No timestamp.npy in {episode_dir}")
            continue

        timestamp = np.load(ts_path)
        label = row["label"]
        start_progress = label.get("start_progress", 0.0)
        final_progress = label["final_progress"]
        segments = label["segments"]
        catastrophic_frame = label.get("catastrophic_frame")

        progress = compute_progress(
            timestamp,
            segments,
            final_progress,
            catastrophic_frame,
            start_progress=start_progress,
        )
        reward = np.diff(progress, prepend=0.0)

        progress_path = episode_dir / "progress.npy"
        reward_path = episode_dir / "reward.npy"

        if args.dry_run:
            print(f"[DRY] {episode_dir.name}: len={len(progress)}, "
                  f"progress=[{progress[0]:.3f}...{progress[-1]:.3f}], "
                  f"final_progress={final_progress}, segments={len(segments)}")
        else:
            np.save(progress_path, progress)
            np.save(reward_path, reward)
            print(f"[OK] {episode_dir.name}: saved progress.npy ({len(progress)},) "
                  f"and reward.npy ({len(reward)},) | final={progress[-1]:.3f}")


if __name__ == "__main__":
    main()
