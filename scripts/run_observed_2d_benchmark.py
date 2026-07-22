#!/usr/bin/env python3
"""Run the decisive synthetic observed_2d-only Stage 1 benchmark.

The runner creates bundles without target_xyz, invokes the existing Stage 1 CLI,
and evaluates against hidden synthetic geometry only after each run finishes.
"""
import argparse
import json
import os
import subprocess
import sys

from build_observed_2d_synthetic import build_bundle


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(command, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        print("RUN", " ".join(command))
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("command failed with {}: {}".format(completed.returncode, log_path))


def names(indices):
    return ",".join(str(x) for x in indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--source_model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--gt_delta_path", default="output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt")
    p.add_argument("--foreground_mask_path", default="output/elephant_source_graphdeco/foreground_mask.pt")
    p.add_argument("--target_image_root", default="output/elephant_source_graphdeco/synthetic_known_delta/targets_body_roundness")
    p.add_argument("--novel_source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/blender_perspective_dataset_transparent")
    p.add_argument("--output_dir", default="output/elephant_source_graphdeco/synthetic_observed_2d_benchmark")
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--skip_existing", action="store_true")
    a = p.parse_args()
    python = sys.executable
    os.makedirs(a.output_dir, exist_ok=True)
    full_bundle = os.path.join(a.output_dir, "clean_8", "observed_2d_bundle.pt")
    full_targets = os.path.join(a.output_dir, "clean_8", "targets")
    if not os.path.exists(full_bundle):
        build_bundle(a.source_path, a.source_model_path, a.gt_delta_path, a.foreground_mask_path,
                     a.target_image_root, full_bundle, full_targets, list(range(8)), seed=a.seed,
                     load_iteration=a.load_iteration)

    conditions = [
        ("clean_1", [0], 0.0, 0.0, 1.0),
        ("clean_2", [0, 4], 0.0, 0.0, 1.0),
        ("clean_4", [0, 2, 4, 6], 0.0, 0.0, 1.0),
        ("clean_8", list(range(8)), 0.0, 0.0, 1.0),
        ("noise_small_8", list(range(8)), 0.5, 0.0, 1.0),
        ("noise_moderate_8", list(range(8)), 2.0, 0.0, 1.0),
        ("outlier_5pct_8", list(range(8)), 0.0, 0.05, 1.0),
        ("outlier_10pct_8", list(range(8)), 0.0, 0.10, 1.0),
        ("coverage_70pct_8", list(range(8)), 0.0, 0.0, 0.70),
    ]
    metrics = {}
    for condition, view_indices, noise, outliers, coverage in conditions:
        condition_dir = os.path.join(a.output_dir, condition)
        bundle_path = os.path.join(condition_dir, "observed_2d_bundle.pt")
        targets = os.path.join(condition_dir, "targets")
        delta_path = os.path.join(condition_dir, "mined_delta_observed_2d.pt")
        metric_path = os.path.join(condition_dir, "metrics.json")
        if not os.path.exists(bundle_path):
            build_bundle(a.source_path, a.source_model_path, a.gt_delta_path, a.foreground_mask_path,
                         a.target_image_root, bundle_path, targets, view_indices,
                         noise_std=noise, outlier_rate=outliers, visibility_keep=coverage,
                         seed=a.seed, load_iteration=a.load_iteration)
        if not (a.skip_existing and os.path.exists(metric_path)):
            command = [python, "train_delta_mining.py", "-s", a.source_path,
                       "--model_path", a.source_model_path,
                       "--load_iteration", str(a.load_iteration),
                       "--target_image_root", targets,
                       "--correspondence_path", bundle_path,
                       "--observation_mode", "observed_2d",
                       "--model_path", a.source_model_path,
                       "--iterations", str(a.iterations),
                       "--save_iterations", str(a.iterations),
                       "--max_d_xyz", "0.08", "--max_d_scaling", "0.0",
                       "--disable_d_scaling", "--free_delta_lr", "0.001",
                       "--lambda_corr_2d", "1.0", "--lambda_corr_3d", "0.0",
                       "--lambda_lpips", "0.0", "--lambda_rgb_weak", "0.0",
                       "--lambda_mask", "0.0", "--lambda_delta", "0.0005",
                       "--lambda_smooth", "0.005", "--smooth_sample", "512",
                       "--foreground_mask_path", a.foreground_mask_path,
                       "--save_delta_path", delta_path, "--white_background",
                       "--weak_target", "false", "--freeze_gaussians", "true", "--quiet"]
            run_command(command, os.path.join(condition_dir, "train.log"))
            evaluate = [python, "scripts/evaluate_observed_2d_recovery.py",
                        "--ground_truth_path", a.gt_delta_path,
                        "--recovered_path", delta_path,
                        "--foreground_mask_path", a.foreground_mask_path,
                        "--reference_bundle", full_bundle,
                        "--source_path", a.source_path,
                        "--model_path", a.source_model_path,
                        "--novel_source_path", a.novel_source_path,
                        "--train_view_indices", names(view_indices),
                        "--load_iteration", str(a.load_iteration),
                        "--output_path", metric_path]
            run_command(evaluate, os.path.join(condition_dir, "evaluate.log"))
        with open(metric_path, "r", encoding="utf-8") as handle:
            metrics[condition] = json.load(handle)

    # Independent A/B four-view split, using only observed 2D bundles.
    for condition, view_indices in (("split_A_4", [0, 2, 4, 6]), ("split_B_4", [1, 3, 5, 7])):
        condition_dir = os.path.join(a.output_dir, condition)
        bundle_path = os.path.join(condition_dir, "observed_2d_bundle.pt")
        targets = os.path.join(condition_dir, "targets")
        delta_path = os.path.join(condition_dir, "mined_delta_observed_2d.pt")
        metric_path = os.path.join(condition_dir, "metrics.json")
        if not os.path.exists(bundle_path):
            build_bundle(a.source_path, a.source_model_path, a.gt_delta_path, a.foreground_mask_path,
                         a.target_image_root, bundle_path, targets, view_indices, seed=a.seed,
                         load_iteration=a.load_iteration)
        if not (a.skip_existing and os.path.exists(metric_path)):
            command = [python, "train_delta_mining.py", "-s", a.source_path,
                       "--model_path", a.source_model_path, "--load_iteration", str(a.load_iteration),
                       "--target_image_root", targets, "--correspondence_path", bundle_path,
                       "--observation_mode", "observed_2d", "--iterations", str(a.iterations),
                       "--save_iterations", str(a.iterations), "--max_d_xyz", "0.08",
                       "--max_d_scaling", "0.0", "--disable_d_scaling", "--free_delta_lr", "0.001",
                       "--lambda_corr_2d", "1.0", "--lambda_corr_3d", "0.0",
                       "--lambda_lpips", "0.0", "--lambda_rgb_weak", "0.0", "--lambda_mask", "0.0",
                       "--lambda_delta", "0.0005", "--lambda_smooth", "0.005", "--smooth_sample", "512",
                       "--foreground_mask_path", a.foreground_mask_path, "--save_delta_path", delta_path,
                       "--white_background", "--weak_target", "false", "--freeze_gaussians", "true", "--quiet"]
            run_command(command, os.path.join(condition_dir, "train.log"))
            evaluate = [python, "scripts/evaluate_observed_2d_recovery.py", "--ground_truth_path", a.gt_delta_path,
                        "--recovered_path", delta_path, "--foreground_mask_path", a.foreground_mask_path,
                        "--reference_bundle", full_bundle, "--source_path", a.source_path,
                        "--novel_source_path", a.novel_source_path,
                        "--model_path", a.source_model_path, "--train_view_indices", names(view_indices),
                        "--load_iteration", str(a.load_iteration), "--output_path", metric_path]
            run_command(evaluate, os.path.join(condition_dir, "evaluate.log"))
        with open(metric_path, "r", encoding="utf-8") as handle: metrics[condition] = json.load(handle)

    ab_path = os.path.join(a.output_dir, "ab_consistency.json")
    run_command([python, "scripts/compare_observed_2d_ab.py",
                 "--delta_a", os.path.join(a.output_dir, "split_A_4", "mined_delta_observed_2d.pt"),
                 "--delta_b", os.path.join(a.output_dir, "split_B_4", "mined_delta_observed_2d.pt"),
                 "--foreground_mask_path", a.foreground_mask_path, "--output_path", ab_path],
                os.path.join(a.output_dir, "ab_consistency.log"))
    with open(ab_path, "r", encoding="utf-8") as handle: ab = json.load(handle)
    summary = {"iterations": a.iterations, "conditions": metrics, "ab_consistency": ab,
               "optimizer_observation_mode": "observed_2d", "target_xyz_in_optimizer": False,
               "ground_truth_used_only_for_generation_and_evaluation": True}
    with open(os.path.join(a.output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
