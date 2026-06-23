"""Scan episode directories and remove bad ones.

Removes episodes that:
  1. Are missing progress.npy or reward.npy
  2. Have progress that is always 0 (no meaningful signal)

Lists all episodes to be removed and waits for confirmation before deleting.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Remove episodes missing progress/reward or with all-zero progress")
    parser.add_argument("--root", required=True, help="Dataset root containing episode_* dirs")
    parser.add_argument(
        "--remove-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also remove episodes whose progress is all zeros (default: True; use --no-remove-zero to keep them)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    episodes = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    if not episodes:
        raise SystemExit(f"No episode_* directories found under {root}")

    print(f"Scanning {len(episodes)} episodes in {root}\n")

    missing = []  # (path, reason)
    all_zero = []  # (path, reason)

    for ep in episodes:
        progress_path = ep / "progress.npy"
        reward_path = ep / "reward.npy"

        missing_files = []
        if not progress_path.exists():
            missing_files.append("progress.npy")
        if not reward_path.exists():
            missing_files.append("reward.npy")

        if missing_files:
            missing.append((ep, f"missing {', '.join(missing_files)}"))
            continue

        if args.remove_zero:
            progress = np.load(progress_path)
            if np.all(progress == 0):
                all_zero.append((ep, f"progress is always 0 (len={len(progress)})"))

    to_remove = missing + all_zero

    if not to_remove:
        print(f"All {len(episodes)} episodes are clean. Nothing to remove.")
        return

    print(f"Found {len(to_remove)} episodes to remove:\n")

    if missing:
        print(f"  Missing progress.npy or reward.npy ({len(missing)}):")
        for ep, reason in missing:
            print(f"    {ep.name}: {reason}")

    if all_zero:
        if missing:
            print()
        print(f"  All-zero progress ({len(all_zero)}):")
        for ep, reason in all_zero:
            print(f"    {ep.name}: {reason}")

    print(f"\nRemove {len(to_remove)} episodes? [Enter=yes / n=no] ", end="")
    answer = input().strip().lower()
    if answer == "" or answer == "y":
        for ep, _ in to_remove:
            shutil.rmtree(ep)
            print(f"  Deleted {ep.name}")
        print(f"\nDone. Removed {len(to_remove)} episodes, {len(episodes) - len(to_remove)} remaining.")
    else:
        print("Skipped.")


if __name__ == "__main__":
    main()
