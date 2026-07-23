#!/usr/bin/env python3
"""Recover and factorize saved oracle observed-2d candidate bundles.

Target xyz is never loaded. The synthetic teacher is evaluator-only.
"""
import argparse
import gc
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import recover_xyz_graph_coupled
from correspondence.schema import ObservationBundle
from stage1.template_factorization import factorize_candidates, delta_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--source_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--max_gaussians", type=int, default=4096)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    source, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
    teacher = torch.load(a.teacher, map_location="cpu")
    active_full = teacher.get("synthetic_region_mask", teacher["foreground_mask"]).bool()
    indices = torch.nonzero(active_full).flatten()
    if len(indices) > a.max_gaussians:
        indices = indices[torch.linspace(0, len(indices) - 1, a.max_gaussians).long()]
    source = source[indices]
    by_name = {c.image_name: c for c in cameras}
    paths = sorted([os.path.join(a.bundle_dir, x) for x in os.listdir(a.bundle_dir) if x.startswith("recovered_bundle_") and x.endswith(".pt")])
    recovered = []
    for path in paths:
        bundle = ObservationBundle.load(path, device="cpu")
        selected = [by_name[name] for name in bundle.camera_names]
        target_xy = bundle.target_xy[:, indices]
        visibility = bundle.visibility_2d[:, indices]
        confidence = bundle.confidence_2d[:, indices]
        result = recover_xyz_graph_coupled(source, selected, target_xy, visibility,
                                           confidence, iterations=a.iterations,
                                           graph_lambda=0.01, min_support=2)
        recovered.append(result["d_xyz"])
        del result, bundle, selected, target_xy, visibility, confidence
        gc.collect()
    recovered = torch.stack(recovered)
    torch.save({"deltas": recovered, "d_scaling": torch.zeros_like(recovered),
                "metadata": {"observation_mode": "observed_2d", "target_xyz_used": False,
                             "solver": "cpu_graph_coupled", "bundle_count": len(paths)}},
               os.path.join(a.output_dir, "recovered_candidate_stack.pt"))
    fg = teacher["foreground_mask"].bool()[indices]; active = active_full[indices]
    gt = teacher["d_xyz"].float()[indices]
    methods = factorize_candidates(recovered)
    report = {}
    for name, value in methods.items():
        if torch.is_tensor(value) and value.ndim == 2:
            m = delta_metrics(value, gt, active_mask=active, foreground_mask=fg)
            m["nuisance_leakage"] = float((value[active] - gt[active]).square().sum() / gt[active].square().sum().clamp_min(1e-8))
            report[name] = m
    with open(os.path.join(a.output_dir, "recovered_factorization_metrics.json"), "w") as handle:
        json.dump({"metrics": report, "gaussians_evaluated": int(len(indices)),
                   "source_indices": indices.tolist()}, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
