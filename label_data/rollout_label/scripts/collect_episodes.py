"""Recursively find all episode_* directories under a given path and copy them to <path>_collection/."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect all episode_* dirs into a flat directory")
    parser.add_argument("path", help="Root path to search for episode_* directories")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    out_dir = root.parent / (root.name + "_flat")

    episodes = sorted(p for p in root.rglob("episode_*") if p.is_dir())
    if not episodes:
        raise SystemExit(f"No episode_* directories found under {root}")

    # Check for name collisions
    names = [ep.name for ep in episodes]
    if len(names) != len(set(names)):
        dupes = [n for n in set(names) if names.count(n) > 1]
        print(f"[WARNING] Duplicate episode names found: {dupes}")
        print("Duplicate sources will overwrite earlier copies.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(episodes)} episodes under {root}")
    print(f"Copying to {out_dir}\n")

    for i, ep in enumerate(episodes):
        dst = out_dir / ep.name
        print(f"[{i + 1}/{len(episodes)}] {ep.relative_to(root)} -> {dst.name}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ep, dst)

    print(f"\nDone. {len(episodes)} episodes collected in {out_dir}")


if __name__ == "__main__":
    main()
