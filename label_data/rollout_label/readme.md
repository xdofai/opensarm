# Rollout Episode Labeling

A web-based tool for labeling individual rollout episodes. For each episode you label:

1. **Final progress** (0 to 1) — how much of the task was completed
2. **Time segments** — split the video timeline and label each segment:
   - `Progress (fast)` — making good progress
   - `Progress (slow)` — making progress but slowly
   - `Adjust` — adjusting/correcting without clear progress
   - `Mistake` — actively making an error
3. **Catastrophic failure** (optional) — mark a frame where an unrecoverable failure occurs. This sets final progress to **-100** and all segments after that frame to "catastrophic failure".

## Setup

Requires Python 3.10+. Use uv to sync dependencies:
```bash
cd ~/rollout_label
uv sync
```

## Dataset

The dataset should be a directory containing `episode_*` subdirectories, each with a `top_camera-images-rgb.mp4` file.



## Full Workflow

```bash
# 1. Check dataset integrity
uv run rollout_label/src/check_data.py --root <PATH/TO/DATASET>

# 2. Label episodes in the web UI
uv run rollout_label/run_label.py --task_name <TASK_NAME> --usr alice --mode new --root <PATH/TO/DATASET> --port 8888

# 3. Generate progress.npy and reward.npy from labels
uv run rollout_label/scripts/generate_progress.py --results <PATH/TO/RESULTS>.jsonl

# 4. Clean up bad episodes (missing progress/reward or all-zero progress)
uv run rollout_label/scripts/clean_episodes.py --root <PATH/TO/DATASET>
# or keep all zero episodes
uv run rollout_label/scripts/clean_episodes.py --root <PATH/TO/DATASET> --no-remove-zero

# 5. Generate validation videos to verify
uv run rollout_label/scripts/validate_video.py --results <PATH/TO/RESULTS>.jsonl
```

## Step 1: Check Dataset

Verify dataset integrity before labeling. Checks that all episodes have the expected video files and npy files, and that frame counts are aligned.

```bash
uv run rollout_label/src/check_data.py --root <PATH/TO/DATASET>
```

Reports any missing files or length mismatches. Offers to delete bad episodes interactively.

## Step 2: Label Episodes

`--task_name` is required and sets the `data/<task_name>/` subdirectory that plans, results, and checkpoints are written to. Reuse the same `--task_name` across runs to support resume.

### Start the web server (recommended):
```bash
uv run rollout_label/run_label.py --task_name <TASK_NAME> --usr <NAME> --mode new --root <PATH/TO/DATASET> --port 8888
```
Then open `http://<HOST-NAME>:8888` in your browser.

### Without specifying a user (configure in browser):
```bash
uv run rollout_label/run_label.py --task_name <TASK_NAME> --host 0.0.0.0 --port 8888
```
Enter your user ID and select mode (new/resume) directly in the web interface.

### Multiple users:
Run on different ports for different annotators (share `--task_name` to share the plan/results dir):
```bash
uv run rollout_label/run_label.py --task_name <TASK_NAME> --port 8888  # User 1
uv run rollout_label/run_label.py --task_name <TASK_NAME> --port 8889  # User 2
```

## Usage

1. Load a plan (new or resume)
2. Watch the episode video
3. **Set final progress** using the slider (0-1)
4. **Segment the timeline**: pause the video and click **"Split Here"** (or press `S`) to split at the current time, then label each segment using the dropdown
   - Right-click a segment to merge it with the next one
   - Click a segment in the list to seek the video to that point
5. **(Optional) Mark catastrophic failure**: pause the video at the failure frame and click "Mark Current Frame as Catastrophic"
6. Click **Save & Next** (or `Ctrl+S`)

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| `Space` | Pause / play |
| `A` / `D` | Seek backward / forward 1 second |
| `S` | Split at current time |
| `Ctrl+S` | Save & next |
| `Left` / `Right` | Previous / next episode |
| `[` / `]` | Slower / faster playback |

## Data Output

Annotation data is saved in `~/rollout_label/data`:
- `results/{usr}_{plan_id}.jsonl` — annotation labels (main output)
- `checkpoint/{usr}_{plan_id}.json` — progress checkpoint for resuming
- `plans/{usr}_{plan_id}.json` — the episode ordering plan

### Output format (JSONL)

Each line in the results file:
```json
{
  "ts": "2026-04-09T12:00:00Z",
  "user": "alice",
  "plan_id": "plan_tshirt_pref_300_raw_seed12345",
  "episode_index": 0,
  "episode_id": 42,
  "video_path": "/path/to/episode_42/top_camera-images-rgb.mp4",
  "label": {
    "final_progress": 0.75,
    "segments": [
      {"start": 0.0, "end": 3.5, "label": "progress_fast"},
      {"start": 3.5, "end": 6.2, "label": "adjust"},
      {"start": 6.2, "end": 10.0, "label": "progress_slow"}
    ],
    "catastrophic_frame": null
  }
}
```

When catastrophic failure is marked:
```json
{
  "label": {
    "final_progress": -100,
    "segments": [
      {"start": 0.0, "end": 4.0, "label": "progress_fast"},
      {"start": 4.0, "end": 10.0, "label": "catastrophic"}
    ],
    "catastrophic_frame": 4.0
  }
}
```

## Step 3: Generate progress.npy and reward.npy

After labeling, run the `generate_progress.py` script to produce per-timestep `progress.npy` and `reward.npy` for each episode. These are saved directly into each episode directory alongside `timestamp.npy`.

### How it works

The script constructs a piecewise-linear progress curve based on segment labels and the final progress value:

| Segment label | Slope |
|---------------|-------|
| `progress_fast` | `+2s` |
| `progress_slow` | `+1s` |
| `adjust` | `0` (flat) |
| `mistake` | `-1s` (retrogress) |

The base slope `s` is solved so that integrating the curve from start to end yields exactly `final_progress`. For catastrophic episodes, progress is set to 0 after the catastrophic frame.

`reward.npy` is the per-step delta of `progress.npy` (i.e., `np.diff(progress, prepend=0)`).

Both arrays have the same length as `timestamp.npy`.

### Usage

```bash
# Dry run — preview without writing files
uv run rollout_label/scripts/generate_progress.py \
    --results data/test/results/<usr>_<plan_id>.jsonl \
    --dry-run

# Generate and save
uv run rollout_label/scripts/generate_progress.py \
    --results data/test/results/<usr>_<plan_id>.jsonl
```

### Example output
```
[OK] episode_abc123: saved progress.npy (613,) and reward.npy (613,) | final=1.000
[OK] episode_def456: saved progress.npy (749,) and reward.npy (749,) | final=0.500
[OK] episode_ghi789: saved progress.npy (1404,) and reward.npy (1404,) | final=0.000  # catastrophic
```

## Step 4: Clean Up Bad Episodes

After generating progress, remove episodes that are missing `progress.npy` / `reward.npy` or have all-zero progress (no meaningful signal). The script lists all episodes to be removed and waits for confirmation before deleting.

```bash
# Default: remove both missing-file episodes and all-zero-progress episodes
uv run rollout_label/scripts/clean_episodes.py --root <PATH/TO/DATASET>

# Only remove episodes missing progress.npy / reward.npy; keep all-zero-progress ones
uv run rollout_label/scripts/clean_episodes.py --root <PATH/TO/DATASET> --no-remove-zero
```

The `--remove-zero` / `--no-remove-zero` flag toggles the all-zero-progress check (default: on).

### Example output
```
Scanning 300 episodes in /path/to/dataset

Found 5 episodes to remove:

  Missing progress.npy or reward.npy (2):
    episode_abc123: missing progress.npy, reward.npy
    episode_def456: missing reward.npy

  All-zero progress (3):
    episode_ghi789: progress is always 0 (len=613)
    episode_jkl012: progress is always 0 (len=749)
    episode_mno345: progress is always 0 (len=1404)

Remove 5 episodes? [Enter=yes / n=no]
```

## Step 5: Validation Videos

After generating progress, you can create side-by-side validation videos (top camera + progress curve) to visually verify the labels. Randomly samples episodes from the results JSONL and saves 10x sped-up videos to `results_dir/validate_video/`.

```bash
# Sample 10 random episodes (default)
uv run rollout_label/scripts/validate_video.py \
    --results <PATH/TO/RESULTS>.jsonl

# Custom count and seed
uv run rollout_label/scripts/validate_video.py \
    --results <PATH/TO/RESULTS>.jsonl -n 5 --seed 123
```

Each video shows the original top camera footage on the left and an animated progress plot on the right, with a red dot tracking the current position and the progress value displayed. Videos are 10x speed (frames are skipped, not sped up) for quick review.
