#!/usr/bin/env python3
"""Recover a diagnostic xyz delta from an observed_2d bundle on CPU."""
import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import recover_xyz_from_observations
from correspondence.schema import ObservationBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_path", required=True)
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--min_support", type=int, default=2)
    p.add_argument("--no_propagate", action="store_true")
    p.add_argument("--mode", choices=["point", "silhouette", "hybrid"], default="point")
    a = p.parse_args()
    bundle = ObservationBundle.load(a.bundle_path, device="cpu")
    if bundle.target_xyz is not None:
        raise ValueError("CPU recovery accepts observed_2d only and refuses target_xyz")
    source_xyz, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
    by_name = {c.image_name: c for c in cameras}
    selected = [by_name[name] for name in bundle.camera_names]
    use_points = a.mode in ("point", "hybrid")
    point_visibility = bundle.visibility_2d if use_points else torch.zeros_like(bundle.visibility_2d)
    point_xy = bundle.target_xy if use_points else torch.zeros_like(bundle.target_xy)
    silhouette = bundle.silhouette_observations if a.mode in ("silhouette", "hybrid") else None
    result = recover_xyz_from_observations(
        source_xyz, selected, point_xy, point_visibility,
        bundle.confidence_2d if use_points else None, iterations=a.iterations,
        min_support=a.min_support, propagate=not a.no_propagate,
        silhouette_observations=silhouette, silhouette_weight=1.0,
        point_weight=1.0,
    )
    d_scaling = torch.zeros_like(result["d_xyz"])
    payload = {"d_xyz": result["d_xyz"], "d_scaling": d_scaling,
               "source_xyz": source_xyz, "foreground_mask": result["support_count"] > 0,
               "support_count": result["support_count"],
               "metadata": {"method": "cpu_multiview_reprojection_irls",
                            "observation_mode": "observed_2d", "target_xyz_used": False,
                            "min_support": a.min_support, "recovery_mode": a.mode, "background_delta_exact_zero": True,
                            "d_scaling_exact_zero": True}}
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    torch.save(payload, a.output_path)
    print({"output": a.output_path, "gaussians": int(len(source_xyz)),
           "supported": int((result["support_count"] > 0).sum()),
           "d_xyz_mean_norm": float(result["d_xyz"].norm(dim=-1).mean())})


if __name__ == "__main__":
    main()
