#!/usr/bin/env python3
"""Diagnose historical manually generated target views without GT leakage."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.schema import ObservationBundle
from correspondence.gaussian_visibility import project_points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--source_image_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset/images")
    p.add_argument("--target_image_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual")
    p.add_argument("--foreground_mask_path", default="output/elephant_source_graphdeco/foreground_mask.pt")
    p.add_argument("--out_dir", default="output/elephant_source_graphdeco/historical_image_observation_diagnostic")
    p.add_argument("--load_iteration", type=int, default=30000)
    a = p.parse_args()
    out = Path(a.out_dir)
    aliases = out / "target_aliases"
    aliases.mkdir(parents=True, exist_ok=True)
    _, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
    targets = sorted(Path(a.target_image_root).glob("*.png"))
    if len(targets) != len(cameras):
        raise ValueError("historical target count {} does not equal source camera count {}".format(len(targets), len(cameras)))
    for camera, target in zip(cameras, targets):
        shutil.copy2(target, aliases / (camera.image_name + ".png"))
    command = [sys.executable, "scripts/extract_image_observations.py", "-s", a.source_path,
               "--model_path", a.model_path, "--source_image_root", a.source_image_root,
               "--target_image_root", str(aliases), "--output_bundle", str(out / "observed_2d_bundle.pt"),
               "--foreground_mask_path", a.foreground_mask_path, "--load_iteration", str(a.load_iteration)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "extract.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("extractor failed; see {}".format(out / "extract.log"))
    bundle = ObservationBundle.load(out / "observed_2d_bundle.pt")
    source_xyz, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
    for idx, (name, camera) in enumerate(zip(bundle.camera_names, cameras)):
        source_path = Path(a.source_image_root) / (name + ".png")
        target_path = aliases / (name + ".png")
        source = cv2.imread(str(source_path))
        target = cv2.imread(str(target_path))
        if target.shape[:2] != source.shape[:2]:
            target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
        canvas = np.concatenate([source, target], axis=1)
        xy, _, _ = project_points(source_xyz, camera.full_proj_transform,
                                  camera.image_width, camera.image_height)
        xy = xy.numpy()
        observed = bundle.target_xy[idx].numpy()
        valid = bundle.visibility_2d[idx].numpy()
        for point, match, keep in zip(xy[::32], observed[::32], valid[::32]):
            if not keep:
                continue
            x, y = np.rint(point).astype(int)
            tx, ty = np.rint(match).astype(int)
            if 0 <= x < source.shape[1] and 0 <= y < source.shape[0] and 0 <= tx < target.shape[1] and 0 <= ty < target.shape[0]:
                cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
                cv2.circle(canvas, (source.shape[1] + tx, ty), 2, (0, 0, 255), -1)
                cv2.line(canvas, (x, y), (source.shape[1] + tx, ty), (255, 180, 0), 1)
        cv2.imwrite(str(out / "overlay_{:02d}.png".format(idx + 1)), canvas)
    report = {"status": "diagnostic_only", "mapping": "sorted_target_prefix_to_sorted_source_camera",
              "mapping_warning": "manual target names are not exact camera names; no arbitrary cross-camera mapping was used beyond ordered key8 pack",
              "bundle": str(out / "observed_2d_bundle.pt"),
              "views": len(bundle.camera_names),
              "gaussians": int(bundle.source_xyz.shape[0]),
              "support_coverage": float((bundle.support_count_2d > 0).float().mean()),
              "per_view_coverage": [float(x.float().mean()) for x in bundle.visibility_2d],
              "target_xyz_used": False,
              "stable_style_delta_claim": False}
    (out / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
