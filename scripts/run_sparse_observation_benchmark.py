#!/usr/bin/env python3
"""Sparse observed-2d -> dense fixed-bank recovery benchmark."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import (build_geometry_cache,
                                          recover_xyz_from_observations,
                                          recover_xyz_graph_coupled_cached)
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from stage1.template_factorization import delta_metrics


def evaluate(pred, gt, fg, active, observed):
    report = delta_metrics(pred, gt, active_mask=active, foreground_mask=fg)
    unobserved = active & ~observed
    report["observed_region"] = delta_metrics(pred, gt, active_mask=observed, foreground_mask=fg)["active"]
    report["unobserved_region"] = delta_metrics(pred, gt, active_mask=unobserved, foreground_mask=fg)["active"]
    report["background_energy"] = float(pred[~fg].square().sum())
    report["d_scaling_max"] = 0.0
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--output_dir", default="output/elephant_source_graphdeco/sparse_observation_benchmark")
    p.add_argument("--load_iteration", type=int, default=30000)
    a = p.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    source, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
    base = ObservationBundle.load("output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt", device="cpu")
    selected = [{c.image_name: c for c in cameras}[name] for name in base.camera_names]
    cache = build_geometry_cache(source, selected, knn=8)
    results = []
    for teacher_name in ("body_roundness", "ear_expansion", "trunk_bending"):
        gt_payload = torch.load("output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{}.pt".format(teacher_name), map_location="cpu")
        gt = gt_payload["d_xyz"].float(); fg = gt_payload["foreground_mask"].bool(); active = gt_payload.get("synthetic_region_mask", fg).bool()
        target_xy = torch.stack([project_points(source + gt, c.full_proj_transform, c.image_width, c.image_height)[0] for c in selected])
        for coverage in (0.1, 0.2, 0.4, 0.6, 1.0):
            for seed in (11, 29, 47, 71, 97):
                teacher_seed = {"body_roundness": 101, "ear_expansion": 202, "trunk_bending": 303}[teacher_name]
                gen = torch.Generator().manual_seed(seed + teacher_seed)
                vis = base.visibility_2d.clone()
                keep = torch.rand(vis.shape, generator=gen) < coverage
                vis &= keep
                observed = vis.any(0)
                started = time.perf_counter()
                corrected = recover_xyz_graph_coupled_cached(cache, target_xy, vis, vis.float(), iterations=12,
                    graph_lambda=0.01, min_support=2, foreground_mask=fg, jacobian_refresh=1)
                corrected_report = evaluate(corrected["d_xyz"], gt, fg, active, observed)
                old = recover_xyz_graph_coupled_cached(cache, target_xy, vis, vis.float(), iterations=12,
                    graph_lambda=0.01, min_support=2, foreground_mask=fg, clear_unobserved=True, jacobian_refresh=1)
                old_report = evaluate(old["d_xyz"], gt, fg, active, observed)
                independent = None
                # The scalar baseline is expensive on the full bank; run it
                # on the decisive body coverage levels while graph methods
                # cover all teachers and masks.
                if teacher_name == "body_roundness" and coverage in (0.2, 0.4, 1.0):
                    independent = recover_xyz_from_observations(source, selected, target_xy, vis, vis.float(), iterations=4,
                min_support=2, propagate=False)["d_xyz"]
                independent_report = None if independent is None else evaluate(independent, gt, fg, active, observed)
                results.append({"teacher": teacher_name, "coverage": coverage, "seed": seed,
                    "observed_gaussians": int(observed.sum()), "corrected": corrected_report,
                    "old_clear_unobserved": old_report, "independent": None if independent is None else independent_report,
                    "runtime_seconds": float(time.perf_counter() - started)})
        # Structured missingness branches, one deterministic seed.
        for branch in ("one_full_view_missing", "active_half_missing"):
            vis = base.visibility_2d.clone()
            if branch == "one_full_view_missing": vis[0] = False
            else: vis[:, torch.nonzero(active).flatten()[::2]] = False
            observed = vis.any(0)
            rec = recover_xyz_graph_coupled_cached(cache, target_xy, vis, vis.float(), iterations=12,
                graph_lambda=0.01, foreground_mask=fg, jacobian_refresh=1)
            results.append({"teacher": teacher_name, "branch": branch,
                            "observed_gaussians": int(observed.sum()),
                            "corrected": evaluate(rec["d_xyz"], gt, fg, active, observed),
                            "background_energy": float(rec["d_xyz"][~fg].square().sum()), "d_scaling_max": 0.0})
    payload = {"benchmark": "sparse_observed_2d_dense_recovery", "coverage_levels": [0.1, 0.2, 0.4, 0.6, 1.0],
               "seeds": [11, 29, 47, 71, 97], "graph_k": 8, "results": results,
               "target_xyz_in_optimizer_input": False, "d_scaling_disabled": True}
    with open(os.path.join(a.output_dir, "sparse_benchmark_summary.json"), "w") as handle: json.dump(payload, handle, indent=2)
    print(json.dumps({"results": len(results), "output": os.path.join(a.output_dir, "sparse_benchmark_summary.json")}, indent=2))


if __name__ == "__main__": main()
