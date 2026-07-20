#!/usr/bin/env python3
"""Create a tiny Blender-style dataset for source 3DGS overfit diagnostics."""

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument(
        "--view_indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional zero-based indices into the sorted source frame list.",
    )
    parser.add_argument("--copy_images", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def frame_sort_key(frame):
    return frame.get("file_path", "")


def find_image(dataset_root, file_path):
    stem = os.path.join(dataset_root, file_path)
    root, ext = os.path.splitext(stem)
    candidates = [stem] if ext else []
    candidates.extend(root + image_ext for image_ext in IMAGE_EXTS)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find image for frame file_path='{}'".format(file_path))


def copy_frame_image(dataset_root, out_dir, frame):
    src = find_image(dataset_root, frame["file_path"])
    rel = os.path.relpath(src, dataset_root)
    dst = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    out_dir = os.path.abspath(args.out_dir)
    train_path = os.path.join(dataset_root, "transforms_train.json")
    test_path = os.path.join(dataset_root, "transforms_test.json")
    if not os.path.isfile(train_path):
        raise FileNotFoundError("Missing transforms_train.json under {}".format(dataset_root))

    train_payload = load_json(train_path)
    test_payload = load_json(test_path) if os.path.isfile(test_path) else {}
    common = {k: v for k, v in train_payload.items() if k != "frames"}
    all_frames = list(train_payload.get("frames", [])) + list(test_payload.get("frames", []))
    all_frames = sorted(all_frames, key=frame_sort_key)
    if not all_frames:
        raise RuntimeError("No frames found in {}".format(dataset_root))

    if args.view_indices:
        selected_indices = args.view_indices
    else:
        selected_indices = list(range(min(args.num_views, len(all_frames))))
    selected = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(all_frames):
            raise IndexError("view index {} out of range [0, {})".format(idx, len(all_frames)))
        selected.append(dict(all_frames[idx]))

    os.makedirs(out_dir, exist_ok=True)
    if args.copy_images:
        for frame in selected:
            copy_frame_image(dataset_root, out_dir, frame)
    else:
        # Keep source images in-place with relative symlinks. This keeps subsets small
        # while preserving the Blender/NeRF file_path convention.
        for frame in selected:
            src = find_image(dataset_root, frame["file_path"])
            rel = os.path.relpath(src, dataset_root)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    payload = dict(common)
    payload["frames"] = selected
    write_json(os.path.join(out_dir, "transforms_train.json"), payload)
    write_json(os.path.join(out_dir, "transforms_test.json"), payload)

    metadata = {
        "source_dataset_root": dataset_root,
        "num_source_frames": len(all_frames),
        "selected_indices": selected_indices,
        "selected_file_paths": [frame["file_path"] for frame in selected],
        "copy_images": bool(args.copy_images),
    }
    write_json(os.path.join(out_dir, "overfit_subset_meta.json"), metadata)

    print("Wrote overfit subset:", out_dir)
    print("Selected views:", len(selected))
    for idx, frame in zip(selected_indices, selected):
        print("  [{}] {}".format(idx, frame["file_path"]))


if __name__ == "__main__":
    main()
