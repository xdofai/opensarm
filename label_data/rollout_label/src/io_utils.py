from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TASK_NAME = os.environ.get("ROLLOUT_LABEL_TASK_NAME")
if not _TASK_NAME:
    raise RuntimeError(
        "ROLLOUT_LABEL_TASK_NAME is not set. "
        "Pass --task_name when launching run_label.py; it sets this env var before importing io_utils."
    )

DATA_DIR = PROJECT_ROOT / "data" / _TASK_NAME
PLANS_DIR = DATA_DIR / "plans"
RESULTS_DIR = DATA_DIR / "results"
CKPT_DIR = DATA_DIR / "checkpoint"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


for d in (DATA_DIR, PLANS_DIR, RESULTS_DIR, CKPT_DIR):
    ensure_dir(d)


def utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, d: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
