#!/usr/bin/env python3
"""Create a manually editable semantic-anchor correspondence template."""
import argparse
import json
import os


ANCHOR_NAMES = [
    "head_center", "body_center", "left_ear_root", "right_ear_root",
    "trunk_root", "trunk_middle", "trunk_tip", "front_left_foot",
    "front_right_foot", "rear_left_foot", "rear_right_foot", "base_center",
    "base_min_x", "base_max_x",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--source_xyz_path", default=None)
    p.add_argument("--target_xyz_path", default=None)
    a = p.parse_args()
    anchors = []
    for name in ANCHOR_NAMES:
        anchors.append({"name": name, "source_xyz": None, "target_xyz": None,
                        "confidence": 0.0, "visible_views": [], "notes": "manual correction required"})
    payload = {"schema": "semantic_anchor_v1", "anchors": anchors,
               "source_xyz_path": os.path.abspath(a.source_xyz_path) if a.source_xyz_path else None,
               "target_xyz_path": os.path.abspath(a.target_xyz_path) if a.target_xyz_path else None,
               "warning": "Anchors are not automatically validated; edit and inspect before dense matching."}
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    print("wrote editable anchor template:", a.output_path)


if __name__ == "__main__": main()
