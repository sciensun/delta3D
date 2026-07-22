#!/usr/bin/env python3
"""Run image-derived matching and fixed-bank recovery on synthetic teachers."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)))
    with log_path.open("w", encoding="utf-8") as f:
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("command failed; see {}".format(log_path))


def subset_targets(source_root, target_root, out_root, names):
    out_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = Path(target_root) / (Path(name).stem + ".png")
        if not source.exists():
            matches = list(Path(target_root).glob(Path(name).stem + ".*"))
            if not matches:
                raise FileNotFoundError("target image missing for {}".format(name))
            source = matches[0]
        shutil.copy2(source, out_root / source.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--source_model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--source_image_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset/images")
    p.add_argument("--foreground_mask_path", default="output/elephant_source_graphdeco/foreground_mask.pt")
    p.add_argument("--output_root", default="output/elephant_source_graphdeco/synthetic_image_observation_benchmark")
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--skip_existing", action="store_true")
    a = p.parse_args()
    gpu_available = torch.cuda.is_available()
    python = sys.executable
    teachers = {
        "body_roundness": "output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt",
        "ear_expansion": "output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_ear_expansion.pt",
        "trunk_bending": "output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_trunk_bending.pt",
    }
    targets = {name: "output/elephant_source_graphdeco/synthetic_known_delta/targets_{}".format(name)
               for name in teachers}
    conditions = [("clean_8", list(range(8)), {})]
    # Controlled appearance/coverage diagnostics are evaluated at the matching
    # layer; the body teacher also runs through Stage 1 for the main gate.
    body_robust = [
        ("brightness_contrast_8", list(range(8)), {"brightness": 18.0, "contrast": 1.15}),
        ("blur_noise_8", list(range(8)), {"blur_sigma": 1.0, "noise_std": 3.0}),
        ("mask_eroded_8", list(range(8)), {"mask_erode": 2}),
        ("split_A_4", [0, 2, 4, 6], {}),
        ("split_B_4", [1, 3, 5, 7], {}),
    ]
    summary = {"backend": "opencv_farneback", "teachers": {}, "oracle_leakage": False}
    for teacher, gt_path in teachers.items():
        summary["teachers"][teacher] = {}
        current_conditions = conditions + body_robust if teacher == "body_roundness" else conditions
        for condition, view_indices, perturb in current_conditions:
            run_dir = Path(a.output_root) / teacher / condition
            bundle = run_dir / "observed_2d_bundle.pt"
            match_metrics = run_dir / "image_observation_metrics.json"
            if not (a.skip_existing and bundle.exists() and match_metrics.exists()):
                extract = [python, "scripts/extract_image_observations.py", "-s", a.source_path,
                           "--model_path", a.source_model_path, "--source_image_root", a.source_image_root,
                           "--target_image_root", targets[teacher], "--output_bundle", str(bundle),
                           "--foreground_mask_path", a.foreground_mask_path,
                           "--load_iteration", str(a.load_iteration), "--view_indices", ",".join(map(str, view_indices)),
                           "--brightness", str(perturb.get("brightness", 0.0)),
                           "--contrast", str(perturb.get("contrast", 1.0)),
                           "--noise_std", str(perturb.get("noise_std", 0.0)),
                           "--blur_sigma", str(perturb.get("blur_sigma", 0.0)),
                           "--mask_erode", str(perturb.get("mask_erode", 0))]
                run(extract, run_dir / "extract.log")
                evaluate = [python, "scripts/evaluate_image_observations.py",
                             "--bundle_path", str(bundle), "--gt_delta_path", gt_path,
                             "-s", a.source_path, "--model_path", a.source_model_path,
                             "--foreground_mask_path", a.foreground_mask_path,
                             "--load_iteration", str(a.load_iteration), "--output_path", str(match_metrics)]
                run(evaluate, run_dir / "evaluate_observations.log")
            with match_metrics.open("r", encoding="utf-8") as f:
                summary["teachers"][teacher][condition] = {"observation": json.load(f)}

            # Only clean teacher runs are used for the recovery gate. The
            # robust appearance conditions remain matcher diagnostics.
            if condition != "clean_8" or not gpu_available:
                if condition == "clean_8" and not gpu_available:
                    summary["teachers"][teacher][condition]["recovery"] = {
                        "status": "not_run_cuda_unavailable"
                    }
                continue
            payload = torch_load(bundle)
            camera_names = payload["camera_names"]
            target_subset = run_dir / "targets"
            subset_targets(a.source_image_root, targets[teacher], target_subset, camera_names)
            delta_path = run_dir / "mined_delta_image_observed_2d.pt"
            if not (a.skip_existing and delta_path.exists()):
                train = [python, "train_delta_mining.py", "-s", a.source_path,
                         "--model_path", a.source_model_path, "--load_iteration", str(a.load_iteration),
                         "--target_image_root", str(target_subset), "--correspondence_path", str(bundle),
                         "--observation_mode", "observed_2d", "--iterations", str(a.iterations),
                         "--max_d_xyz", "0.08", "--max_d_scaling", "0.0", "--disable_d_scaling",
                         "--lambda_corr_2d", "1.0", "--lambda_corr_3d", "0.0",
                         "--lambda_lpips", "0.0", "--lambda_rgb_weak", "0.0", "--lambda_mask", "0.0",
                         "--lambda_delta", "0.0005", "--lambda_smooth", "0.005", "--smooth_sample", "512",
                         "--foreground_mask_path", a.foreground_mask_path, "--save_delta_path", str(delta_path),
                         "--white_background", "--weak_target", "false", "--freeze_gaussians", "true", "--quiet"]
                run(train, run_dir / "train.log")
            recovered_metrics = run_dir / "recovery_metrics.json"
            evaluate_recovery = [python, "scripts/evaluate_observed_2d_recovery.py",
                                 "--ground_truth_path", gt_path, "--recovered_path", str(delta_path),
                                 "--foreground_mask_path", a.foreground_mask_path,
                                 "--reference_bundle", str(bundle), "--source_path", a.source_path,
                                 "--novel_source_path", "assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent",
                                 "--model_path", a.source_model_path, "--train_view_indices", ",".join(map(str, view_indices)),
                                 "--load_iteration", str(a.load_iteration), "--output_path", str(recovered_metrics)]
            run(evaluate_recovery, run_dir / "evaluate_recovery.log")
            with recovered_metrics.open("r", encoding="utf-8") as f:
                summary["teachers"][teacher][condition]["recovery"] = json.load(f)
    with (Path(a.output_root) / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
