#!/usr/bin/env python3
"""Collect manually generated ChatGPT key8 results and prepare a Tripo input image."""

import argparse
import json
import os
import shutil
from datetime import datetime


DEFAULT_REPO_ROOT = "/home/shichang/Deformable-3D-Gaussians"
DEFAULT_OBJECT_ID = "big_carved_wooden_elephant_sculpture"
DEFAULT_PREPARED = os.path.join(DEFAULT_REPO_ROOT, "assets/prepared", DEFAULT_OBJECT_ID)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_dir", default=os.path.join(DEFAULT_PREPARED, "generated_standard/key8_manual"))
    parser.add_argument("--prompts_json", default=os.path.join(DEFAULT_PREPARED, "prompts/prompts_standard_key8.json"))
    parser.add_argument("--tripo_dir", default=os.path.join(DEFAULT_PREPARED, "tripo_input"))
    parser.add_argument("--report", default=os.path.join(DEFAULT_PREPARED, "generated_standard/manual_collection_report.json"))
    return parser.parse_args()


def find_generated(manual_dir, index):
    prefixes = ["{:02d}_standard".format(index), "{:02d}".format(index)]
    for prefix in prefixes:
        for ext in IMAGE_EXTENSIONS:
            path = os.path.join(manual_dir, prefix + ext)
            if os.path.isfile(path):
                return path
    if os.path.isdir(manual_dir):
        for name in sorted(os.listdir(manual_dir)):
            lower = name.lower()
            if lower.startswith("{:02d}_".format(index)) and lower.endswith(IMAGE_EXTENSIONS):
                return os.path.join(manual_dir, name)
    return None


def load_prompt_items(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])


def choose_front_threequarter(found, prompt_items):
    if 2 in found:
        return 2, found[2], "preferred numeric prefix 02"

    for item in prompt_items:
        azimuth = int(round(float(item.get("azimuth", -1)))) % 360
        idx = int(item.get("view_index", -1)) + 1
        if azimuth == 45 and idx in found:
            return idx, found[idx], "preferred azimuth 045"

    if found:
        idx = sorted(found)[0]
        return idx, found[idx], "front-three-quarter result missing; using first available"
    return None, None, "no generated manual results found"


def main():
    args = parse_args()
    os.makedirs(args.manual_dir, exist_ok=True)
    os.makedirs(args.tripo_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    prompt_items = load_prompt_items(args.prompts_json)
    found = {}
    missing = []
    for idx in range(1, 9):
        path = find_generated(args.manual_dir, idx)
        if path:
            found[idx] = path
        else:
            missing.append("{:02d}_standard.png".format(idx))

    chosen_idx, chosen_path, reason = choose_front_threequarter(found, prompt_items)
    tripo_path = None
    if chosen_path:
        tripo_path = os.path.join(args.tripo_dir, "standard_front_3quarter.png")
        shutil.copy2(chosen_path, tripo_path)

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "manual_dir": os.path.abspath(args.manual_dir),
        "expected_outputs": ["{:02d}_standard.png".format(i) for i in range(1, 9)],
        "found": {str(k): v for k, v in sorted(found.items())},
        "missing": missing,
        "chosen_index": chosen_idx,
        "chosen_source": chosen_path,
        "chosen_reason": reason,
        "tripo_input": tripo_path,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Manual collection report:", args.report)
    if missing:
        print("Missing manual results:", ", ".join(missing))
    if tripo_path:
        print("Tripo-ready input:", tripo_path)
    else:
        print("No Tripo-ready input created.")


if __name__ == "__main__":
    main()
