#!/usr/bin/env python3
"""Prepare empty manifests for conditional target-template variants."""
import argparse
import json
from pathlib import Path


PROMPT = """Convert this rendered stylized wooden elephant sculpture into a standard, typical, ordinary elephant sculpture.

Preserve the exact requested camera/view and compatible framing, elephant category, corresponding semantic parts, compatible topology and number of major parts, and broadly compatible pose. Keep the trunk direction, ear/leg placement, base attachment, and overall object arrangement compatible. Reduce blockiness and faceting moderately and make volumes smoother, more rounded, and anatomically coherent. Do not add or delete major parts, make an extreme pose change, change the camera, or introduce unrelated stylization. Aim to preserve body build, ear proportions, trunk proportions, limb proportions, overall coloration, and surface appearance where possible. Minor natural generation variation is acceptable. Output one centered object only with the same framing and no text or extra objects.
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
    target_names = ["sample_A", "sample_B", "sample_C"]
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
                                 "requires_semantic_part_check": True, "template_is_conditional_sample": True},
            "metadata": {"generation_status": "not_generated", "observation_status": "not_extracted",
                         "generation_condition": "same_standardized_prompt",
                         "observed_appearance_attributes": None,
                         "observed_geometry_attributes": None,
                         "posthoc_nuisance_labels": None},
        })
    root.mkdir(parents=True, exist_ok=True)
    (root / "style_task_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    template_records = []
    for record in records:
        template_records.append({
            "target_template_id": record["object_id"] + ":" + record["repeat_id"],
            "target_style_family": record["style_family"], "style_operation": "blocky_to_rounded",
            "style_intensity": record["intensity"], "template_variant_id": record["repeat_id"],
            "target_style_attributes": record["target_attributes"],
            "template_nuisance_attributes": {},
            "appearance_nuisance_attributes": None,
            "geometry_nuisance_attributes": None,
            "required_invariants": ["elephant category", "main parts", "compatible topology", "broad pose", "camera/view"],
            "allowed_variations": ["color", "surface appearance", "moderate body build", "moderate part proportions"],
            "forbidden_changes": ["added/deleted major parts", "extreme pose change", "camera change"],
            "view_relation": {"same_camera_names": names},
            "semantic_part_requirements": ["head", "trunk", "ears", "legs", "body", "base"],
            "generation_seed": None, "generation_run": "not_generated",
            "quality_metadata": record["quality_control"], "object_id": record["object_id"],
        })
    (root / "target_template_manifest.json").write_text(json.dumps(template_records, indent=2), encoding="utf-8")
    (root / "prompt_template.txt").write_text(PROMPT, encoding="utf-8")
    print(json.dumps({"manifest": str(root / "style_task_manifest.json"),
                      "prompt": str(root / "prompt_template.txt"),
                      "repeats": target_names, "cameras": names}, indent=2))


if __name__ == "__main__":
    main()
