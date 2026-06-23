from __future__ import annotations
import csv
import re

import torch
import torch.nn.functional as F

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


@torch.no_grad()
def task_class_match_check(
    task_name_prototype: torch.Tensor,   # (num_tasks, D) or (D,)
    class_prototype: torch.Tensor,       # (num_class, D) or (D,)
    task_list: list[str],
    class_list: list[str],
    temperature: float = 1.0,
    eps: float = 1e-8,
    distance: str = "l2",                # "l2" or "cosine"
    normalize: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    print_matrix: bool = True,
) -> torch.Tensor:
    """
    Pairwise distance -> inverse -> row-wise softmax.
    Returns probs: (num_tasks, num_class).
    """
    T = task_name_prototype
    C = class_prototype

    if device is not None:
        T = T.to(device)
        C = C.to(device)
    if dtype is not None:
        T = T.to(dtype)
        C = C.to(dtype)

    if T.dim() == 1:
        T = T.unsqueeze(0)   # (1, D)
    if C.dim() == 1:
        C = C.unsqueeze(0)   # (1, D)

    assert T.dim() == 2 and C.dim() == 2, f"Expect 2D, got T{tuple(T.shape)} C{tuple(C.shape)}"
    assert T.size(1) == C.size(1), f"Dim mismatch: T{tuple(T.shape)} vs C{tuple(C.shape)}"

    if normalize:
        T = F.normalize(T, p=2, dim=1, eps=eps)
        C = F.normalize(C, p=2, dim=1, eps=eps)

    if distance == "l2":
        # dist: (num_tasks, num_class)
        dist = torch.cdist(T, C, p=2)
    elif distance == "cosine":
        # dist = 1 - cosine_similarity
        cos = T @ C.t()                    # (num_tasks, num_class)
        dist = 1.0 - cos
    else:
        raise ValueError("distance must be 'l2' or 'cosine'")

    sim = 1.0 / (dist + eps)              # inverse distance as similarity
    logits = sim / max(temperature, eps)
    probs = F.softmax(logits, dim=1)      # row-wise softmax over classes

    if print_matrix:
        print("[INIT] Task-Class Match Distance:")
        # print("probs shape:", tuple(probs.shape))
        # print nicely
        with torch.no_grad():
            print(probs.detach().cpu())
            
        for i in range(len(task_list)):
            print(f"Task [{task_list[i]}] matches class [{class_list[probs[i].argmax().item()]}] with prob {probs[i].max().item():.4f}")

    return probs



    
"""
Aggregate MoE expert load across episodes.

Usage patterns:
1) compute_moe(root/<task_name>)
   - scans episode_* under that task directory
   - writes report to: root/<task_name>/moe_expert_load_report.txt

2) compute_moe(root)
   - scans tasks under root (subdirs containing episode_*)
   - also scans any episode_* directly under root
   - writes:
        root/moe_expert_load_report_GLOBAL.txt
        root/<task_name>/moe_expert_load_report.txt   (per task found)
"""



def _find_episode_dirs(base: Path) -> List[Path]:
    """Find episode_* dirs under base that contain episode_summary.txt."""
    eps: List[Path] = []
    if not base.exists():
        return eps
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("episode_"):
            continue
        try:
            _ = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if (p / "episode_summary.txt").exists():
            eps.append(p)
    return eps


def _load_expert_weights(ep_dir: Path) -> Optional[Dict[str, float]]:
    """
    Parse episode_summary.txt to extract expert_avg_weights block.

    Expected:

      expert_avg_weights:
        expert_0: 0.0103
        expert_1: 0.0352

      top4_experts:
        ...

    Returns {expert_name -> avg_weight}, or None.
    """
    summary_path = ep_dir / "episode_summary.txt"
    if not summary_path.exists():
        return None

    weights: Dict[str, float] = {}
    in_block = False

    try:
        with summary_path.open("r") as f:
            for line in f:
                stripped = line.strip()

                if not in_block:
                    if stripped.startswith("expert_avg_weights:"):
                        in_block = True
                    continue

                # end block on blank line or next section
                if stripped == "" or stripped.startswith("top"):
                    break

                if ":" in stripped:
                    name, val = stripped.split(":", 1)
                    name = name.strip()
                    try:
                        w = float(val.strip())
                    except ValueError:
                        continue
                    weights[name] = w
    except Exception:
        return None

    return weights if weights else None


def _summarize_episode_dirs(episodes: List[Path]) -> Tuple[Dict[str, float], int, int]:
    """
    Returns:
      avg_weights: expert -> avg_weight across episodes (averaged per-episode)
      used_eps: number of episodes with valid weights
      experts_found: number of experts in avg_weights
    """
    sum_w = defaultdict(float)
    cnt_w = defaultdict(int)

    used_eps = 0

    for ep in episodes:
        w_dict = _load_expert_weights(ep)
        if w_dict is None:
            continue
        used_eps += 1
        for name, w in w_dict.items():
            sum_w[name] += w
            cnt_w[name] += 1

    avg_weights: Dict[str, float] = {}
    for name, total in sum_w.items():
        c = cnt_w[name]
        if c > 0:
            avg_weights[name] = total / c

    return avg_weights, used_eps, len(avg_weights)


def _format_report(title: str, avg_weights: Dict[str, float], used_eps: int, top_k: int) -> str:
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    if used_eps == 0 or not avg_weights:
        lines.append("No expert_avg_weights loaded from any episode_summary.txt.")
        return "\n".join(lines) + "\n"

    sorted_items = sorted(avg_weights.items(), key=lambda kv: kv[1], reverse=True)
    lines.append(f"Episodes with valid weights : {used_eps}")
    lines.append(f"Experts found               : {len(sorted_items)}")
    lines.append(f"Top-{top_k} experts by average load:")
    for rank, (name, w) in enumerate(sorted_items[:top_k], start=1):
        lines.append(f"  #{rank}: {name:>10s}  avg_weight = {w:.4f}")

    num_experts = len(sorted_items)
    k = min(top_k, num_experts)
    baseline = 1.0 / num_experts
    topk_mean = sum(w for _, w in sorted_items[:k]) / k
    ratio = topk_mean / baseline
    collapse_ratio = num_experts / k
    collapse_score = (ratio - 1) / (collapse_ratio - 1) if collapse_ratio > 1 else 0.0
    lines.append(
        f"Top-{k} concentration ratio   : {ratio:.4f}  "
        f"(baseline = 1.0 [uniform], collapse = {collapse_ratio:.4f} [routes only to top-{k}])"
    )
    lines.append(
        f"Top-{k} collapse score        : {collapse_score:.4f}  "
        f"(0 = uniform, 1 = fully collapsed onto top-{k}; N/k-agnostic)"
    )
    return "\n".join(lines) + "\n"


def compute_moe(path, top_k: int):
    """
    If `path` is root/<task_name>:
      - scan episode_* inside that folder
      - save report under <task_name>/moe_expert_load_report.txt
      - return a dict: {"mode":"single_task", "task":..., "report_path":..., "avg_weights":..., "used_eps":...}

    If `path` is root/:
      - treat each immediate subdir as a "task" if it contains episode_*
      - also scan episode_* directly under root as a special task "__root__" (if any)
      - write per-task reports under each task folder
      - write a global merged report under root/moe_expert_load_report_GLOBAL.txt
      - return a dict summary for all tasks
    """
    base = Path(os.path.expanduser(str(path))).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    # Heuristic: is this a task directory?
    # If base itself contains episode_* dirs => treat as single-task path.
    base_eps = _find_episode_dirs(base)
    has_subtask_dirs = any(p.is_dir() and _find_episode_dirs(p) for p in base.iterdir())

    # --------- Case A: base is root/<task_name> (single task) ----------
    if base_eps and not has_subtask_dirs:
        avg_w, used_eps, _ = _summarize_episode_dirs(base_eps)
        report_txt = _format_report(
            title=f"MoE Expert Load Report (task='{base.name}')",
            avg_weights=avg_w,
            used_eps=used_eps,
            top_k=top_k,
        )
        report_path = base / "moe_expert_load_report.txt"
        report_path.write_text(report_txt)

        print(f"[OK] Wrote: {report_path}")
        return {
            "mode": "single_task",
            "task": base.name,
            "report_path": str(report_path),
            "avg_weights": avg_w,
            "used_eps": used_eps,
        }

    # --------- Case B: base is root/ containing tasks ----------
    tasks: Dict[str, List[Path]] = {}

    # 1) direct episodes under root (optional)
    if base_eps:
        tasks["__root__"] = base_eps

    # 2) each subdir that contains episode_* is a task
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        eps = _find_episode_dirs(sub)
        if eps:
            tasks[sub.name] = eps

    if not tasks:
        # nothing to do
        report_path = base / "moe_expert_load_report_GLOBAL.txt"
        report_path.write_text(
            _format_report(
                title="MoE Expert Load Report (GLOBAL)",
                avg_weights={},
                used_eps=0,
                top_k=top_k,
            )
        )
        print(f"[WARN] No episode_* dirs with episode_summary.txt found under: {base}")
        print(f"[OK] Wrote empty global report: {report_path}")
        return {"mode": "root", "root": str(base), "tasks": {}, "global_report_path": str(report_path)}

    # Per-task reports + collect for global merge
    global_sum = defaultdict(float)
    global_cnt = defaultdict(int)
    global_used_eps = 0

    task_summaries = {}

    for task_name, eps in tasks.items():
        avg_w, used_eps, _ = _summarize_episode_dirs(eps)

        # write per-task report
        if task_name == "__root__":
            task_dir = base
            report_path = base / "moe_expert_load_report__root__.txt"
            title = "MoE Expert Load Report (task='__root__')"
        else:
            task_dir = base / task_name
            report_path = task_dir / "moe_expert_load_report.txt"
            title = f"MoE Expert Load Report (task='{task_name}')"

        report_path.write_text(_format_report(title=title, avg_weights=avg_w, used_eps=used_eps, top_k=top_k))
        print(f"[OK] Wrote: {report_path}")

        task_summaries[task_name] = {
            "task_dir": str(task_dir),
            "report_path": str(report_path),
            "used_eps": used_eps,
            "avg_weights": avg_w,
        }

        # merge into global using the same averaging scheme as before:
        # (sum per-expert over episodes / count per-expert over episodes)
        # We can reconstruct this merge robustly by re-reading per-episode weights again.
        # But to keep it simple + consistent, we merge from episodes directly:
        for ep in eps:
            w_dict = _load_expert_weights(ep)
            if w_dict is None:
                continue
            global_used_eps += 1
            for name, w in w_dict.items():
                global_sum[name] += w
                global_cnt[name] += 1

    global_avg = {name: (global_sum[name] / global_cnt[name]) for name in global_sum if global_cnt[name] > 0}

    global_report_path = base / "moe_expert_load_report_GLOBAL.txt"
    global_report_path.write_text(
        _format_report(
            title="MoE Expert Load Report (GLOBAL)",
            avg_weights=global_avg,
            used_eps=global_used_eps,
            top_k=top_k,
        )
    )
    print(f"[OK] Wrote: {global_report_path}")

    return {
        "mode": "root",
        "root": str(base),
        "tasks": task_summaries,
        "global_report_path": str(global_report_path),
        "global_used_eps": global_used_eps,
        "global_avg_weights": global_avg,
    }


_TOPK_LINE_RE = re.compile(r"^\s*#\d+:\s*expert_(\d+)\b", re.IGNORECASE)


def _parse_topk_experts(report_path: Path, top_k: int = 4) -> Optional[List[int]]:
    """
    Parse moe_expert_load_report.txt and return top-k expert indices, e.g. [0,7,1,5].
    """
    try:
        txt = report_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    indices: List[int] = []
    for line in txt.splitlines():
        m = _TOPK_LINE_RE.match(line)
        if m:
            indices.append(int(m.group(1)))
            if len(indices) >= top_k:
                break

    if len(indices) == 0:
        return None
    return indices



def collect_task_expert_topk_to_csv(
    eval_folder: str | Path, 
    top_k: int,
    task_sequence_list: Optional[List[str]]
) -> Path:
    """
    Collects Top-K MoE expert indices and saves them to a CSV file.
    Rows:
      1. Task names (ordered by task_sequence_list if provided)
      2. Top-K indices formatted as a string like "[0 7 1 5]"
    """
    eval_folder = Path(eval_folder).expanduser().resolve()
    out_csv = eval_folder / "expert_topk_summary.csv"

    # Dictionary to store results: { task_name: formatted_topk_string }
    results_cache: Dict[str, str] = {}

    # 1. Scan and parse
    for task_dir in eval_folder.iterdir():
        if not task_dir.is_dir():
            continue

        report_path = task_dir / "moe_expert_load_report.txt"
        if not report_path.exists():
            continue

        topk_indices = _parse_topk_experts(report_path, top_k=top_k)
        if topk_indices is not None:
            # Format list of ints into string: "[0 7 1 5]"
            formatted_str = "[" + " ".join(map(str, topk_indices)) + "]"
            results_cache[task_dir.name] = formatted_str

    # 2. Determine column order
    if task_sequence_list is not None:
        ordered_tasks = [t for t in task_sequence_list if t in results_cache]
    else:
        ordered_tasks = sorted(results_cache.keys())

    # 3. Extract data row
    topk_row = [results_cache[t] for t in ordered_tasks]

    # 4. Write to CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ordered_tasks)
        writer.writerow(topk_row)

    return out_csv

