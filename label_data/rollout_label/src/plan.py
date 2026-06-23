from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from src.io_utils import PLANS_DIR, write_json

TOP_MP4 = "top_camera-images-rgb.mp4"


@dataclass
class Episode:
    video_path: str
    episode_id: int


@dataclass
class Plan:
    plan_id: str
    root_dir: str
    n_items: int
    episodes: List[Episode]


def _find_candidates(root: Path) -> List[Path]:
    cands = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            mp4 = child / TOP_MP4
            if mp4.exists():
                cands.append(child)
    return sorted(cands)


def make_plan(root_dir: str, seed: int = 0, plan_id: str | None = None) -> Plan:
    root = Path(root_dir)
    items = _find_candidates(root)
    if not items:
        raise RuntimeError(f"No candidates found under {root_dir} with required file: {TOP_MP4}")

    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    episodes: List[Episode] = []
    for eid, idx in enumerate(indices):
        episodes.append(Episode(video_path=str(items[idx] / TOP_MP4), episode_id=eid))

    plan_id = plan_id or f"plan_{root.name}_seed{seed}"
    return Plan(plan_id=plan_id, root_dir=str(root), n_items=len(items), episodes=episodes)


def save_plan(plan: Plan, usr: str) -> Path:
    path = PLANS_DIR / f"{usr}_{plan.plan_id}.json"
    write_json(
        path,
        {
            "plan_id": plan.plan_id,
            "root_dir": plan.root_dir,
            "n_items": plan.n_items,
            "episodes": [asdict(e) for e in plan.episodes],
        },
    )
    return path
