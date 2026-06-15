#!/usr/bin/env python3
"""Convert rendered key8/full36 PNGs plus views_meta.json into a minimal Blender-style dataset."""

import argparse
import json
import math
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views_meta", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--camera_distance", type=float, default=4.0)
    parser.add_argument("--camera_angle_x", type=float, default=0.691111)
    return parser.parse_args()


def camera_to_world(azimuth, elevation, distance):
    az = math.radians(azimuth)
    el = math.radians(elevation)
    loc = [
        distance * math.sin(az) * math.cos(el),
        -distance * math.cos(az) * math.cos(el),
        distance * math.sin(el),
    ]
    # Minimal approximate camera transform. This helper warns because the source
    # renders are orthographic but the repo's Blender loader expects perspective.
    return [
        [1.0, 0.0, 0.0, loc[0]],
        [0.0, 1.0, 0.0, loc[1]],
        [0.0, 0.0, 1.0, loc[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main():
    args = parse_args()
    with open(args.views_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    src_dir = os.path.dirname(os.path.abspath(args.views_meta))
    train_dir = os.path.join(args.out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    frames = []
    n = max(len(meta["views"]) - 1, 1)
    for idx, view in enumerate(meta["views"]):
        src = view.get("path") or os.path.join(src_dir, view["filename"])
        dst_name = os.path.basename(view["filename"])
        dst = os.path.join(train_dir, dst_name)
        shutil.copy2(src, dst)
        stem = os.path.splitext(dst_name)[0]
        frames.append(
            {
                "file_path": "train/" + stem,
                "time": idx / n,
                "transform_matrix": camera_to_world(view["azimuth"], view["elevation"], args.camera_distance),
            }
        )

    transforms = {
        "camera_angle_x": args.camera_angle_x,
        "warning": "Source renders were orthographic. This approximate dataset uses perspective camera_angle_x because the repo Blender loader does not support orthographic cameras.",
        "frames": frames,
    }
    for name in ("transforms_train.json", "transforms_test.json"):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=2)
    print("Wrote approximate Blender dataset:", args.out_dir)
    print("WARNING: orthographic source cameras were approximated as perspective.")


if __name__ == "__main__":
    main()
