#!/usr/bin/env python3
"""Prepare empty, explicit manifests for the first real repeated pilot."""
import argparse
import json
from pathlib import Path


PROMPT = """Convert this rendered stylized wooden elephant sculpture into a moderately rounded, less blocky version.

Preserve the exact same camera viewpoint, crop, object identity, pose, trunk direction, ear placement, leg placement, body placement, base attachment, topology, and number of parts. Keep the same material, color, lighting, and background as far as possible. Reduce blockiness and faceting moderately and make the volumes smoother, more rounded, and anatomically coherent. Do not add or delete parts, change posture, change the camera, redesign the texture, or introduce unrelated realism changes. Output one centered object only with the same framing and no text or extra objects.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="assets/prepared/big_carved_wooden_elephant_sculpture")
    p.add_argument("--source_image_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset/images")
    p.add_argument("--source_glb", default="assets/3D/big_carved_wooden_elephant_sculpture.glb")
    p.add_argument("--source_3dgs", default="output/elephant_source_graphdeco")
    a = p.parse_args()
    root = Path(a.root) / "real_pilot_blocky_to_rounded"
    names = sorted(p.name for p in Path(a.source_image_root).glob("*.png"))
    target_names = ["repeat_{}".format(i) for i in range(3)]
    records = []
    for repeat in target_names:
        target_root = root / repeat / "targets_key8"
        target_root.mkdir(parents=True, exist_ok=True)
        records.append({
            "object_id": "big_carved_wooden_elephant_sculpture",
            "object_category": "wooden elephant sculpture",
            "source_glb": a.source_glb,
            "source_3dgs": a.source_3dgs,
            "style_family": "blocky_to_rounded",
            "source_attributes": {"blockiness": "high", "faceting": "high"},
            "target_attributes": {"blockiness": "moderately reduced", "roundness": "increased"},
            "intensity": 0.5,
            "repeat_id": repeat,
            "camera_names": names,
            "source_image_root": a.source_image_root,
            "target_image_root": str(target_root),
            "generation_prompt": PROMPT,
            "affected_parts": ["body surface", "body volume"],
            "preserved_parts": ["head", "trunk", "ears", "legs", "base", "pose", "camera", "topology"],
            "quality_control": {"required_complete_views": len(names), "requires_manual_silhouette_check": True,
                                 "requires_identity_check": True, "repeat_is_independent": True},
            "metadata": {"generation_status": "not_generated", "observation_status": "not_extracted"},
        })
    root.mkdir(parents=True, exist_ok=True)
    (root / "style_task_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (root / "prompt_template.txt").write_text(PROMPT, encoding="utf-8")
    print(json.dumps({"manifest": str(root / "style_task_manifest.json"),
                      "prompt": str(root / "prompt_template.txt"),
                      "repeats": target_names, "cameras": names}, indent=2))


if __name__ == "__main__":
    main()
