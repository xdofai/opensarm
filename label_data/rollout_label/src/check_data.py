import argparse
import os
import shutil
import struct
import numpy as np
VIDEO_NAMES = ["top_camera-images-rgb.mp4", "left_camera-images-rgb.mp4", "right_camera-images-rgb.mp4"]
NPY_NAMES = [
    "action-left-pos.npy",
    "action-right-pos.npy",
    "left-gripper_pos.npy",
    "right-gripper_pos.npy",
    "left-joint_pos.npy",
    "right-joint_pos.npy",
    "timestamp.npy",
]


def get_video_frame_count(path):
    """Get frame count by parsing the mp4 stsz atom."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size, box_type = struct.unpack(">I4s", header)
            box_type = box_type.decode("ascii", errors="replace")
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                header_size = 16
            else:
                header_size = 8
            if size == 0:
                break
            if box_type in ("moov", "trak", "mdia", "minf", "stbl"):
                continue
            elif box_type == "stsz":
                f.read(4)  # version + flags
                sample_size = struct.unpack(">I", f.read(4))[0]
                sample_count = struct.unpack(">I", f.read(4))[0]
                return sample_count
            else:
                f.seek(size - header_size, 1)
    return None


def get_video_duration(path):
    """Get mp4 duration by parsing the moov/mvhd atom (no external deps)."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size, box_type = struct.unpack(">I4s", header)
            box_type = box_type.decode("ascii", errors="replace")
            if size == 1:  # 64-bit extended size
                size = struct.unpack(">Q", f.read(8))[0]
                header_size = 16
            else:
                header_size = 8
            if size == 0:
                break

            if box_type in ("moov", "trak", "mdia"):
                # container box — descend into children
                continue
            elif box_type == "mvhd":
                version = struct.unpack(">B", f.read(1))[0]
                f.read(3)  # flags
                if version == 0:
                    f.read(8)  # creation_time + modification_time (4+4)
                    timescale = struct.unpack(">I", f.read(4))[0]
                    duration = struct.unpack(">I", f.read(4))[0]
                else:
                    f.read(16)  # creation_time + modification_time (8+8)
                    timescale = struct.unpack(">I", f.read(4))[0]
                    duration = struct.unpack(">Q", f.read(8))[0]
                return duration / timescale
            else:
                # skip this box
                f.seek(size - header_size, 1)
    raise ValueError(f"Could not find mvhd atom in {path}")


def main():
    parser = argparse.ArgumentParser(description="Check dataset integrity: videos, npy files, frame alignment")
    parser.add_argument("--root", required=True, help="Dataset root containing episode_* dirs")
    args = parser.parse_args()

    BASE_DIR = args.root
    episodes = sorted(
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("episode_")
    )

    print(f"Found {len(episodes)} episodes")

    bad_episodes = []  # list of (ep_name, list_of_issues)

    for ep in episodes:
        ep_path = os.path.join(BASE_DIR, ep)
        files = os.listdir(ep_path)
        issues = []

        # Check videos and get frame count reference
        frame_counts = []
        for video_name in VIDEO_NAMES:
            video_path = os.path.join(ep_path, video_name)
            if video_name in files:
                try:
                    fc = get_video_frame_count(video_path)
                    if fc is not None:
                        frame_counts.append(fc)
                except Exception as e:
                    issues.append(f"{video_name}: ERROR - {e}")
            else:
                issues.append(f"{video_name}: MISSING")

        # Check npy files
        ref_frame_count = frame_counts[0] if frame_counts else None
        for npy_name in NPY_NAMES:
            npy_path = os.path.join(ep_path, npy_name)
            if npy_name in files:
                try:
                    arr = np.load(npy_path)
                    if ref_frame_count is not None and len(arr) != ref_frame_count:
                        issues.append(f"{npy_name}: length {len(arr)} vs {ref_frame_count} video frames")
                except Exception as e:
                    issues.append(f"{npy_name}: ERROR - {e}")
            else:
                issues.append(f"{npy_name}: MISSING")

        if issues:
            bad_episodes.append((ep, issues))

    # Summary
    print(f"\n{len(episodes) - len(bad_episodes)}/{len(episodes)} episodes OK")
    if bad_episodes:
        print(f"{len(bad_episodes)} episodes with issues:\n")
        for ep, issues in bad_episodes:
            print(f"  {ep}:")
            for issue in issues:
                print(f"    - {issue}")
        # Ask to delete
        print(f"\nDelete {len(bad_episodes)} bad episodes? [Enter=yes / n=no] ", end="")
        answer = input().strip().lower()
        if answer == "" or answer == "y":
            for ep, _ in bad_episodes:
                ep_path = os.path.join(BASE_DIR, ep)
                shutil.rmtree(ep_path)
                print(f"  Deleted {ep}")
            print("Done.")
        else:
            print("Skipped.")
    else:
        print("All episodes are clean!")


if __name__ == "__main__":
    main()
