import os
import re
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Set, Optional
from collections import defaultdict

import numpy as np
import torch


def adapt_lerobot_batch_sarm(
    batch: Dict[str, Any],
    camera_names: List[str] = ["top_camera-images-rgb"],
    eval_video: bool = False,
) -> Dict[str, Any]:
    """
    Convert a batch to the (multi-stage) LeRobot-compatible format.

    When eval_video=True, wrap single-example tensors with a leading batch dim
    and wrap the scalar task into a single-item list to mimic batched inputs.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": maybe_unsqueeze(batch["lengths"]),
        "tasks": [batch["task"]] if eval_video else batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result


def adapt_lerobot_batch_rewind(
    batch: dict,
    camera_names: List[str] = ["top_camera-images-rgb"],
    eval_video: bool = False
) -> dict:
    """Convert to lerobot-compatible batch format.
    
    Args:
        batch: Input batch dictionary.
        camera_names: List of camera keys to include.
        eval_video: If True, wrap tensors with an additional batch dimension.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": maybe_unsqueeze(batch["lengths"]),
        "tasks": batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result


def adapt_lerobot_batch_act_pri(
    batch: dict,
    camera_names: List[str] = ["top_camera-images-rgb"],
    n_obs_steps: int = 4,
    eval_video: bool = False
) -> dict:
    """Convert to lerobot-compatible batch format.
    
    Args:
        batch: Input batch dictionary.
        camera_names: List of camera keys to include.
        dense_annotation: Whether to transpose task annotations.
        eval_video: If True, wrap tensors with an additional batch dimension.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": maybe_unsqueeze(batch["lengths"]),
        "steps_to_go": maybe_unsqueeze(batch["steps_to_go"]),
        "tasks": batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    if "act_pri_index" in batch:
        result["act_pri_index"] = maybe_unsqueeze(batch["act_pri_index"])[:, 1:n_obs_steps + 1] 
        # skip the first frame, it is the start of the episode
    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result


def get_valid_episodes(repo_id: str) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.

    Args:
        repo_id (str): HuggingFace repo ID, 

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    episode_pattern = re.compile(r"episode_(\d+)\.parquet")

    valid_episodes = []

    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for file in chunk_dir.glob("episode_*.parquet"):
            match = episode_pattern.match(file.name)
            if match:
                ep_idx = int(match.group(1))
                valid_episodes.append(ep_idx)

    return sorted(valid_episodes)

def split_train_eval_episodes(valid_episodes: List[int], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episodes into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        train_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes



# ============================================================================
# Episode-selection & evaluation analytics (ported from sarm/data_utils.py)
# ============================================================================
def find_eps_each_task(
    num_per_task: int,
    val_eps: List[int],
    repo_id: str,
    task_list: List[str],
    seed: int,
) -> Dict[str, List[int]]:
    """
    For each task in task_list, pick `num_per_task` episode_index.

    Priority:
      1) episodes whose episode_index is in `val_eps`
      2) if not enough, fill with episodes for that task NOT in `val_eps` (and print warning)

    Reads:
      ~/.cache/huggingface/lerobot/<repo_id>/meta/episodes.jsonl

    Returns:
      dict: {task_name: [episode_index, ...]} (length == num_per_task when possible)
    """
    eps_path = (
        Path("~/.cache/huggingface/lerobot")
        .expanduser()
        / repo_id
        / "meta"
        / "episodes.jsonl"
    )
    if not eps_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {eps_path}")

    # ---- load and index episodes by task ----
    by_task: Dict[str, List[int]] = {t: [] for t in task_list}
    with eps_path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {line_no} in {eps_path}: {e}") from e

            ep = obj.get("episode_index", None)
            tasks = obj.get("tasks", None)

            if ep is None or tasks is None:
                continue
            if not isinstance(tasks, list):
                continue

            # each episode can potentially have multiple tasks; add to any requested task
            for t in tasks:
                if t in by_task:
                    by_task[t].append(int(ep))

    val_set = set(int(x) for x in val_eps)
    rng = random.Random(int(seed))

    out: Dict[str, List[int]] = {}

    for task in task_list:
        all_eps = list(dict.fromkeys(by_task.get(task, [])))  # de-dup, keep order
        if len(all_eps) == 0:
            # task not present in this dataset (e.g. "dummy" placeholder); skip it
            print(f"[WARN] task='{task}': no episodes found in {eps_path}, skipping")
            continue
        in_val = [e for e in all_eps if e in val_set]
        not_in_val = [e for e in all_eps if e not in val_set]

        picked: List[int] = []

        if len(in_val) >= num_per_task:
            picked = rng.sample(in_val, k=num_per_task)
        else:
            picked = list(in_val)
            need = num_per_task - len(picked)
            print(f"[WARN] task='{task}': only {len(in_val)}/{num_per_task} found")

           
        out[task] = picked

    return out





def _find_episode_dirs_with_gt(base: Path, excluded_names: Set[str], excluded_idxs: Set[int]) -> List[Path]:
    """Find episode_* directories under base that contain gt.npy, excluding specified episodes."""
    eps: List[Path] = []
    if not base.exists():
        return eps

    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("episode_"):
            continue
        if not (p / "gt.npy").exists():
            continue

        # skip by folder name
        if p.name in excluded_names:
            continue

        # parse idx + skip by idx
        try:
            idx = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if idx in excluded_idxs:
            continue

        eps.append(p)
    return eps


def _load_pair(ep_dir: Path, pred_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load {pred_name}.npy and gt.npy from an episode directory.

    Returns:
      (pred, gt) as float64 1D arrays with only finite entries,
      truncated to same length; or None on failure.
    """
    pred_path = ep_dir / f"{pred_name}.npy"
    gt_path = ep_dir / "gt.npy"
    if not (pred_path.exists() and gt_path.exists()):
        return None
    try:
        pred = np.load(pred_path, allow_pickle=False)
        gt = np.load(gt_path, allow_pickle=False)

        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        gt = np.asarray(gt, dtype=np.float64).reshape(-1)

        L = min(len(pred), len(gt))
        if L == 0:
            return None
        pred = pred[:L]
        gt = gt[:L]

        mask = np.isfinite(pred) & np.isfinite(gt)
        if not np.any(mask):
            return None
        return pred[mask], gt[mask]
    except Exception:
        return None


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def _format_report(title: str,
                   per_ep: List[Tuple[str, float, int]],
                   macro: float,
                   micro: float,
                   pred_name: str,
                   excluded_idxs: Set[int],
                   excluded_names: Set[str]) -> str:
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"pred file          : {pred_name}.npy")
    lines.append(f"gt file            : gt.npy")
    lines.append(f"excluded idxs      : {sorted(excluded_idxs) if excluded_idxs else []}")
    lines.append(f"excluded names     : {sorted(excluded_names) if excluded_names else []}")
    lines.append("")
    if not per_ep:
        lines.append("No valid episodes to aggregate.")
        return "\n".join(lines) + "\n"

    lines.append(f"Episodes counted                : {len(per_ep)}")
    lines.append(f"Macro-average MSE (mean of ep)  : {macro:.6f}")
    lines.append(f"Micro-average MSE (all samples) : {micro:.6f}")
    lines.append("")
    
    return "\n".join(lines) + "\n"


def compute_mse(
    path,
    pred_name: str = "smoothed",               # or "pred"
    excluded_episodes: Optional[Set[int]] = None,
    report_name: str = "mse_report.txt",
    verbose: bool = True,
) -> Dict:
    """
    Compute MSE between `{pred_name}.npy` and `gt.npy` across episodes.

    Behavior:
    - If `path` is a leaf task dir (contains episode_*), scans only that dir and writes:
        <path>/<report_name>
    - If `path` is a root dir, scans:
        - episode_* directly under root (bucket "__root__")
        - episode_* under each immediate subdir (bucket = subdir name)
      Writes:
        <root>/<report_name>                     (GLOBAL)
        <root>/<task>/<report_name>              (per task bucket)
        <root>/<report_name with __root__>       (if root has episodes)

    Returns a dict with global + per-bucket stats.
    """
    base = Path(os.path.expanduser(str(path))).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    excluded_idxs = set(excluded_episodes or set())
    excluded_names = {f"episode_{i:06d}" for i in excluded_idxs}

    # Detect if base is a single task dir: it has episodes, and it doesn't look like a root with multiple task subdirs
    base_eps = _find_episode_dirs_with_gt(base, excluded_names, excluded_idxs)
    has_subtask_eps = any(p.is_dir() and _find_episode_dirs_with_gt(p, excluded_names, excluded_idxs) for p in base.iterdir())

    def summarize_episode_dirs(eps: List[Path]) -> Tuple[List[Tuple[str, float, int]], float, float, float, int]:
        per_ep: List[Tuple[str, float, int]] = []
        total_sse = 0.0
        total_cnt = 0

        for ep in eps:
            pair = _load_pair(ep, pred_name=pred_name)
            if pair is None:
                continue
            pred, gt = pair
            n = len(pred)
            if n == 0:
                continue

            ep_mse = _mse(pred, gt)
            per_ep.append((ep.name, ep_mse, n))

            diff = pred - gt
            total_sse += float(np.sum(diff * diff))
            total_cnt += n

        if not per_ep:
            return per_ep, float("nan"), float("nan"), 0.0, 0

        macro = float(np.mean([m for _, m, _ in per_ep]))
        micro = (total_sse / total_cnt) if total_cnt > 0 else float("nan")
        return per_ep, macro, micro, total_sse, total_cnt

    # -------- Case A: single task dir --------
    if base_eps and not has_subtask_eps:
        per_ep, macro, micro, _, _ = summarize_episode_dirs(base_eps)
        report_path = base / report_name
        report_path.write_text(
            _format_report(
                title=f"MSE Report (task='{base.name}')",
                per_ep=per_ep,
                macro=macro if np.isfinite(macro) else float("nan"),
                micro=micro if np.isfinite(micro) else float("nan"),
                pred_name=pred_name,
                excluded_idxs=excluded_idxs,
                excluded_names=excluded_names,
            )
        )
        if verbose:
            print(f"[OK] Wrote: {report_path}")
            if per_ep:
                print(f"  Episodes counted: {len(per_ep)}  macro={macro:.6f}  micro={micro:.6f}")
            else:
                print("  No valid episodes.")

        return {
            "mode": "single_task",
            "path": str(base),
            "task": base.name,
            "report_path": str(report_path),
            "episodes_counted": len(per_ep),
            "macro_mse": macro,
            "micro_mse": micro,
        }

    # -------- Case B: root dir (multi-task buckets) --------
    buckets: Dict[str, List[Path]] = {}

    if base_eps:
        buckets["__root__"] = base_eps

    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        eps = _find_episode_dirs_with_gt(sub, excluded_names, excluded_idxs)
        if eps:
            buckets[sub.name] = eps

    if not buckets:
        # Write empty global report
        global_report_path = base / report_name
        global_report_path.write_text(
            _format_report(
                title="MSE Report (GLOBAL)",
                per_ep=[],
                macro=float("nan"),
                micro=float("nan"),
                pred_name=pred_name,
                excluded_idxs=excluded_idxs,
                excluded_names=excluded_names,
            )
        )
        if verbose:
            print(f"[WARN] No episodes with gt.npy found under: {base}")
            print(f"[OK] Wrote empty global report: {global_report_path}")
        return {"mode": "root", "path": str(base), "buckets": {}, "global_report_path": str(global_report_path)}

    # Per-bucket stats + global merge (micro via total_sse/total_cnt; macro via mean of per-episode MSEs)
    bucket_stats = {}
    all_per_ep: List[Tuple[str, float, int]] = []
    global_total_sse = 0.0
    global_total_cnt = 0

    for name, eps in buckets.items():
        per_ep, macro, micro, sse, cnt = summarize_episode_dirs(eps)

        # write per-bucket report
        if name == "__root__":
            report_path = base / report_name.replace(".txt", "__root__.txt")
            title = "MSE Report (bucket='__root__')"
        else:
            report_path = (base / name) / report_name
            title = f"MSE Report (task='{name}')"

        report_path.write_text(
            _format_report(
                title=title,
                per_ep=per_ep,
                macro=macro if np.isfinite(macro) else float("nan"),
                micro=micro if np.isfinite(micro) else float("nan"),
                pred_name=pred_name,
                excluded_idxs=excluded_idxs,
                excluded_names=excluded_names,
            )
        )
        if verbose:
            print(f"[OK] Wrote: {report_path}")

        bucket_stats[name] = {
            "report_path": str(report_path),
            "episodes_counted": len(per_ep),
            "macro_mse": macro,
            "micro_mse": micro,
        }

        # merge into global
        all_per_ep.extend([(f"{name}/{ep}", m, n) for ep, m, n in per_ep])
        global_total_sse += sse
        global_total_cnt += cnt

    if all_per_ep:
        global_macro = float(np.mean([m for _, m, _ in all_per_ep]))
        global_micro = (global_total_sse / global_total_cnt) if global_total_cnt > 0 else float("nan")
    else:
        global_macro = float("nan")
        global_micro = float("nan")

    global_report_path = base / report_name
    global_report_path.write_text(
        _format_report(
            title="MSE Report (GLOBAL)",
            per_ep=[(ep, m, n) for ep, m, n in all_per_ep],
            macro=global_macro,
            micro=global_micro,
            pred_name=pred_name,
            excluded_idxs=excluded_idxs,
            excluded_names=excluded_names,
        )
    )
    if verbose:
        print(f"[OK] Wrote: {global_report_path}")
        if all_per_ep:
            print(f"  GLOBAL episodes counted: {len(all_per_ep)}  macro={global_macro:.6f}  micro={global_micro:.6f}")
        else:
            print("  GLOBAL: no valid episodes.")

    return {
        "mode": "root",
        "path": str(base),
        "global_report_path": str(global_report_path),
        "global_episodes_counted": len(all_per_ep),
        "global_macro_mse": global_macro,
        "global_micro_mse": global_micro,
        "buckets": bucket_stats,
    }

def _load_act_pri_pair(ep_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load pred_act_pri.npy and gt_act_pri.npy from an episode dir."""
    pred_path = ep_dir / "pred_act_pri.npy"
    gt_path = ep_dir / "gt_act_pri.npy"
    if not (pred_path.exists() and gt_path.exists()):
        return None
    try:
        pred = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.int64).reshape(-1)
        gt = np.asarray(np.load(gt_path, allow_pickle=False), dtype=np.int64).reshape(-1)
        L = min(len(pred), len(gt))
        if L == 0:
            return None
        return pred[:L], gt[:L]
    except Exception:
        return None


def _find_episode_dirs_with_act_pri(base: Path) -> List[Path]:
    eps: List[Path] = []
    if not base.exists():
        return eps
    for p in sorted(base.iterdir()):
        if not p.is_dir() or not p.name.startswith("episode_"):
            continue
        if not (p / "pred_act_pri.npy").exists() or not (p / "gt_act_pri.npy").exists():
            continue
        eps.append(p)
    return eps


def _format_act_pri_report(title: str,
                           per_ep: List[Tuple[str, Dict[str, Any]]],
                           agg: Dict[str, float],
                           dummy_task_idx: int) -> str:
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"dummy task index   : {dummy_task_idx}")
    lines.append("")
    if not per_ep:
        lines.append("No valid episodes to aggregate.")
        return "\n".join(lines) + "\n"

    lines.append(f"Episodes counted                       : {len(per_ep)}")
    lines.append(f"ActPri  accuracy (vanilla)             : {agg['task_acc_vanilla']:.6f}   N={agg['n_vanilla']}")
    lines.append(f"ActPri  accuracy (no-dummy GT)         : {agg['task_acc_nodummy']:.6f}   N={agg['n_nodummy']}")
    lines.append(f"Class   accuracy (vanilla)             : {agg['class_acc_vanilla']:.6f}   N={agg['n_vanilla']}")
    lines.append(f"Class   accuracy (no-dummy GT)         : {agg['class_acc_nodummy']:.6f}   N={agg['n_nodummy']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def compute_act_pri_accuracy(
    path,
    task_to_class_id,
    dummy_task_idx: int,
    report_name: str = "act_pri_accuracy_report.txt",
    verbose: bool = True,
) -> Dict:
    """
    Compute act_pri (task) and class estimation accuracy across episodes.

    Reads pred_act_pri.npy / gt_act_pri.npy under episode_* dirs.
    Reports both vanilla accuracy and the variant that excludes samples
    whose GT task index equals `dummy_task_idx`.

    Behavior mirrors compute_mse:
    - If `path` is a leaf task dir, writes <path>/<report_name>.
    - If `path` is a root dir, writes per-task and a GLOBAL report.
    """
    base = Path(os.path.expanduser(str(path))).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    if isinstance(task_to_class_id, torch.Tensor):
        t2c = task_to_class_id.detach().cpu().numpy().astype(np.int64)
    else:
        t2c = np.asarray(task_to_class_id, dtype=np.int64)

    def _per_ep_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
        pred_c = t2c[pred]
        gt_c = t2c[gt]
        task_ok = (pred == gt)
        class_ok = (pred_c == gt_c)
        mask = (gt != dummy_task_idx)
        n = int(len(gt))
        n_nd = int(mask.sum())
        return {
            "n": n,
            "n_nd": n_nd,
            "task_correct": int(task_ok.sum()),
            "class_correct": int(class_ok.sum()),
            "task_correct_nd": int((task_ok & mask).sum()),
            "class_correct_nd": int((class_ok & mask).sum()),
            "task_acc": float(task_ok.mean()) if n > 0 else float("nan"),
            "class_acc": float(class_ok.mean()) if n > 0 else float("nan"),
            "task_acc_nd": float((task_ok & mask).sum() / n_nd) if n_nd > 0 else float("nan"),
            "class_acc_nd": float((class_ok & mask).sum() / n_nd) if n_nd > 0 else float("nan"),
        }

    def _aggregate(per_ep: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, float]:
        if not per_ep:
            nan = float("nan")
            return {
                "task_acc_vanilla": nan, "class_acc_vanilla": nan,
                "task_acc_nodummy": nan, "class_acc_nodummy": nan,
                "n_vanilla": 0, "n_nodummy": 0,
            }
        # micro accuracy: sum correct / sum count across episodes
        total_n = sum(s["n"] for _, s in per_ep)
        total_n_nd = sum(s["n_nd"] for _, s in per_ep)
        task_acc_v = (sum(s["task_correct"] for _, s in per_ep) / total_n) if total_n > 0 else float("nan")
        class_acc_v = (sum(s["class_correct"] for _, s in per_ep) / total_n) if total_n > 0 else float("nan")
        task_acc_nd = (sum(s["task_correct_nd"] for _, s in per_ep) / total_n_nd) if total_n_nd > 0 else float("nan")
        class_acc_nd = (sum(s["class_correct_nd"] for _, s in per_ep) / total_n_nd) if total_n_nd > 0 else float("nan")
        return {
            "task_acc_vanilla": task_acc_v, "class_acc_vanilla": class_acc_v,
            "task_acc_nodummy": task_acc_nd, "class_acc_nodummy": class_acc_nd,
            "n_vanilla": total_n, "n_nodummy": total_n_nd,
        }

    def summarize_episode_dirs(eps: List[Path]) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, float]]:
        per_ep: List[Tuple[str, Dict[str, Any]]] = []
        for ep in eps:
            pair = _load_act_pri_pair(ep)
            if pair is None:
                continue
            pred, gt = pair
            per_ep.append((ep.name, _per_ep_stats(pred, gt)))
        return per_ep, _aggregate(per_ep)

    base_eps = _find_episode_dirs_with_act_pri(base)
    has_subtask_eps = any(
        p.is_dir() and _find_episode_dirs_with_act_pri(p) for p in base.iterdir()
    )

    # -------- Case A: single task dir --------
    if base_eps and not has_subtask_eps:
        per_ep, agg = summarize_episode_dirs(base_eps)
        report_path = base / report_name
        report_path.write_text(
            _format_act_pri_report(
                title=f"ActPri Accuracy Report (task='{base.name}')",
                per_ep=per_ep,
                agg=agg,
                dummy_task_idx=dummy_task_idx,
            )
        )
        if verbose:
            print(f"[OK] Wrote: {report_path}")
        return {
            "mode": "single_task",
            "path": str(base),
            "task": base.name,
            "report_path": str(report_path),
            "episodes_counted": len(per_ep),
            **agg,
        }

    # -------- Case B: root dir (multi-task buckets) --------
    buckets: Dict[str, List[Path]] = {}
    if base_eps:
        buckets["__root__"] = base_eps
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        eps = _find_episode_dirs_with_act_pri(sub)
        if eps:
            buckets[sub.name] = eps

    if not buckets:
        global_report_path = base / report_name
        global_report_path.write_text(
            _format_act_pri_report(
                title="ActPri Accuracy Report (GLOBAL)",
                per_ep=[],
                agg=_aggregate([]),
                dummy_task_idx=dummy_task_idx,
            )
        )
        if verbose:
            print(f"[WARN] No episodes with pred/gt_act_pri.npy under: {base}")
        return {"mode": "root", "path": str(base), "buckets": {}, "global_report_path": str(global_report_path)}

    bucket_stats: Dict[str, Any] = {}
    all_per_ep: List[Tuple[str, Dict[str, Any]]] = []

    for name, eps in buckets.items():
        per_ep, agg = summarize_episode_dirs(eps)
        if name == "__root__":
            report_path = base / report_name.replace(".txt", "__root__.txt")
            title = "ActPri Accuracy Report (bucket='__root__')"
        else:
            report_path = (base / name) / report_name
            title = f"ActPri Accuracy Report (task='{name}')"

        report_path.write_text(
            _format_act_pri_report(
                title=title,
                per_ep=per_ep,
                agg=agg,
                dummy_task_idx=dummy_task_idx,
            )
        )
        if verbose:
            print(f"[OK] Wrote: {report_path}")
        bucket_stats[name] = {"report_path": str(report_path), "episodes_counted": len(per_ep), **agg}
        all_per_ep.extend([(f"{name}/{ep}", s) for ep, s in per_ep])

    global_agg = _aggregate(all_per_ep)
    global_report_path = base / report_name
    global_report_path.write_text(
        _format_act_pri_report(
            title="ActPri Accuracy Report (GLOBAL)",
            per_ep=all_per_ep,
            agg=global_agg,
            dummy_task_idx=dummy_task_idx,
        )
    )
    if verbose:
        print(f"[OK] Wrote: {global_report_path}")

    return {
        "mode": "root",
        "path": str(base),
        "global_report_path": str(global_report_path),
        "global_episodes_counted": len(all_per_ep),
        "buckets": bucket_stats,
        **{f"global_{k}": v for k, v in global_agg.items()},
    }


_RE_ACT_PRI_V = re.compile(r"^ActPri\s+accuracy \(vanilla\)\s*:\s*([0-9.eE+-]+)")
_RE_ACT_PRI_ND = re.compile(r"^ActPri\s+accuracy \(no-dummy GT\)\s*:\s*([0-9.eE+-]+)")
_RE_CLASS_V = re.compile(r"^Class\s+accuracy \(vanilla\)\s*:\s*([0-9.eE+-]+)")
_RE_CLASS_ND = re.compile(r"^Class\s+accuracy \(no-dummy GT\)\s*:\s*([0-9.eE+-]+)")


def _parse_act_pri_report(txt_path: Path) -> Optional[Dict[str, float]]:
    out: Dict[str, float] = {}
    patterns = {
        "act_pri_vanilla": _RE_ACT_PRI_V,
        "act_pri_nodummy": _RE_ACT_PRI_ND,
        "class_vanilla": _RE_CLASS_V,
        "class_nodummy": _RE_CLASS_ND,
    }
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        for key, regex in patterns.items():
            if key in out:
                continue
            m = regex.match(line)
            if m:
                out[key] = float(m.group(1))
                break
    return out if out else None


def collect_task_act_pri_accuracy_to_csv(
    eval_folder: str | Path,
    task_sequence_list: Optional[List[str]] = None,
) -> Path:
    """
    Walks `eval_folder/<task>/act_pri_accuracy_report.txt`, collects the 4 accuracy
    numbers per task, and writes a CSV with 5 rows:
      row 0: task names
      row 1: ActPri accuracy (vanilla)
      row 2: ActPri accuracy (no-dummy GT)
      row 3: Class  accuracy (vanilla)
      row 4: Class  accuracy (no-dummy GT)
    """
    eval_folder = Path(eval_folder).expanduser().resolve()
    out_csv = eval_folder / "act_pri_accuracy_summary.csv"

    temp_data: Dict[str, Dict[str, float]] = {}
    for task_dir in eval_folder.iterdir():
        if not task_dir.is_dir():
            continue
        report_path = task_dir / "act_pri_accuracy_report.txt"
        if not report_path.exists():
            continue
        parsed = _parse_act_pri_report(report_path)
        if parsed:
            temp_data[task_dir.name] = parsed

    if task_sequence_list is not None:
        ordered_tasks = [t for t in task_sequence_list if t in temp_data]
    else:
        ordered_tasks = sorted(temp_data.keys())

    nan = float("nan")
    tasks: List[str] = []
    act_pri_v: List[float] = []
    act_pri_nd: List[float] = []
    class_v: List[float] = []
    class_nd: List[float] = []
    for t in ordered_tasks:
        tasks.append(t)
        d = temp_data[t]
        act_pri_v.append(d.get("act_pri_vanilla", nan))
        act_pri_nd.append(d.get("act_pri_nodummy", nan))
        class_v.append(d.get("class_vanilla", nan))
        class_nd.append(d.get("class_nodummy", nan))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(tasks)
        w.writerow(act_pri_v)
        w.writerow(act_pri_nd)
        w.writerow(class_v)
        w.writerow(class_nd)

    return out_csv


_RE_EPISODES = re.compile(r"^Episodes counted\s*:\s*(\d+)\s*$")
_RE_MACRO = re.compile(r"^Macro-average MSE .*:\s*([0-9.eE+-]+)\s*$")
_RE_MICRO = re.compile(r"^Micro-average MSE .*:\s*([0-9.eE+-]+)\s*$")


def _parse_mse_report(txt_path: Path) -> Optional[Dict[str, Any]]:
    episodes = None
    macro = None
    micro = None

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        m = _RE_EPISODES.match(line)
        if m:
            episodes = int(m.group(1))
            continue
        m = _RE_MACRO.match(line)
        if m:
            macro = float(m.group(1))
            continue
        m = _RE_MICRO.match(line)
        if m:
            micro = float(m.group(1))
            continue

    if macro is None and micro is None and episodes is None:
        return None

    return {
        "episodes": episodes,
        "macro_mse": macro,
        "micro_mse": micro,
        "report_path": str(txt_path),
    }



def collect_task_mse_to_csv(eval_folder: str | Path, 
                            task_sequence_list: Optional[List[str]]) -> Path:
    """
    Output CSV with exactly 3 rows.
    If task_sequence_list is provided, rows will follow that specific task order.
    """
    eval_folder = Path(eval_folder).expanduser().resolve()
    out_csv = eval_folder / "mse_summary.csv"

    # Use a dictionary to store parsed data: { task_name: {"macro": float, "micro": float} }
    temp_data: Dict[str, Dict[str, float]] = {}

    # 1. Collect all available data from the folder
    for task_dir in eval_folder.iterdir():
        if not task_dir.is_dir():
            continue
            
        report_path = task_dir / "mse_report.txt"
        if not report_path.exists():
            continue

        parsed = _parse_mse_report(report_path)
        if parsed:
            temp_data[task_dir.name] = {
                "macro_mse": parsed["macro_mse"],
                "micro_mse": parsed["micro_mse"]
            }

    # 2. Determine the final order
    if task_sequence_list is not None:
        # Use the provided sequence, but only include tasks that actually exist in temp_data
        ordered_tasks = [t for t in task_sequence_list if t in temp_data]
    else:
        # Default: alphanumeric sort of the found tasks
        ordered_tasks = sorted(temp_data.keys())

    # 3. Prepare rows for CSV
    tasks: List[str] = []
    macro_mse: List[float] = []
    micro_mse: List[float] = []

    for t_name in ordered_tasks:
        tasks.append(t_name)
        macro_mse.append(temp_data[t_name]["macro_mse"])
        micro_mse.append(temp_data[t_name]["micro_mse"])

    # 4. Write to CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(tasks)
        w.writerow(macro_mse)
        w.writerow(micro_mse)

    return out_csv
