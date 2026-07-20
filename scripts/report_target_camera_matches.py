#!/usr/bin/env python3
"""Report source camera to weak target image matches for Stage 1."""

import argparse
import json
import os
from collections import Counter
from types import SimpleNamespace

from PIL import Image

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.style_image_utils import find_style_target_path


EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--target_image_root", required=True)
    parser.add_argument("--max_report", type=int, default=20)
    parser.add_argument("--include_test", action="store_true")
    return parser.parse_args()


def load_frames(source_path, include_test=False):
    frames = []
    for split, filename in (("train", "transforms_train.json"), ("test", "transforms_test.json")):
        if split == "test" and not include_test:
            continue
        path = os.path.join(source_path, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, frame in enumerate(data.get("frames", [])):
            image_name = os.path.basename(frame.get("file_path", ""))
            frames.append(SimpleNamespace(uid=len(frames), image_name=image_name, split=split, frame_index=idx))
    return frames


def count_target_images(root):
    total = 0
    names = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(EXTS):
                total += 1
                names.append(os.path.relpath(os.path.join(dirpath, filename), root))
    return total, sorted(names)


def main():
    args = parse_args()
    frames = load_frames(args.source_path, include_test=args.include_test)
    target_count, target_names = count_target_images(args.target_image_root)
    matches = []
    missing = []
    used_targets = Counter()
    for frame in frames:
        path = find_style_target_path(frame, args.target_image_root, required=False)
        if path:
            rel = os.path.relpath(path, args.target_image_root)
            matches.append((frame, path))
            used_targets[rel] += 1
        else:
            missing.append(frame)

    duplicate_targets = {name: count for name, count in used_targets.items() if count > 1}
    fallback_sorted_order = False

    print("source dataset:", args.source_path)
    print("target root:", args.target_image_root)
    print("source cameras considered:", len(frames))
    print("target images found:", target_count)
    print("matched views:", len(matches))
    print("missing views:", len(missing))
    print("fallback sorted-order matching used:", fallback_sorted_order)
    if duplicate_targets:
        print("WARNING: target images matched by multiple source cameras:")
        for name, count in sorted(duplicate_targets.items()):
            print("  {} <- {} cameras".format(name, count))
    print("")
    print("Target files:")
    for name in target_names[: args.max_report]:
        print("  " + name)
    if len(target_names) > args.max_report:
        print("  ... {} more".format(len(target_names) - args.max_report))
    print("")
    print("Matched pairs:")
    for frame, path in matches[: args.max_report]:
        print("  [{}] {} -> {}".format(frame.split, frame.image_name, path))
    if len(matches) > args.max_report:
        print("  ... {} more".format(len(matches) - args.max_report))
    if missing:
        print("")
        print("Missing examples:")
        for frame in missing[: args.max_report]:
            print("  [{}] {}".format(frame.split, frame.image_name))

    if duplicate_targets:
        raise SystemExit("Ambiguous matching: at least one target maps to multiple source cameras.")
    if not matches:
        raise SystemExit("No target-camera matches found.")


if __name__ == "__main__":
    main()
