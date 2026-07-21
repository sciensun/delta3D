#!/usr/bin/env python3
"""Create controlled xyz-only deformation teachers on the canonical Gaussian bank."""
import argparse
import json
import os

import torch
from plyfile import PlyData


def load_xyz(model_path, iteration):
    path = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    vertex = PlyData.read(path)["vertex"].data
    xyz = torch.stack([torch.from_numpy(vertex[n].copy()) for n in ("x", "y", "z")], dim=1).float()
    return xyz, path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_d_xyz", type=float, default=0.08)
    p.add_argument("--deformations", nargs="+", default=["body_roundness", "ear_expansion", "trunk_bending"])
    return p.parse_args()


def q(values, value):
    return torch.quantile(values, torch.tensor(float(value), dtype=values.dtype))


def make_deformation(xyz, foreground, name, max_d):
    lo, hi = xyz[foreground].min(0).values, xyz[foreground].max(0).values
    center = (lo + hi) / 2.0
    half = (hi - lo).clamp_min(1e-6) / 2.0
    p = (xyz - center) / half
    y = p[:, 1]
    x = p[:, 0]
    z = p[:, 2]
    d = torch.zeros_like(xyz)

    if name == "body_roundness":
        # Central torso proxy: expand radially in the horizontal x/z plane.
        region = foreground & (y > -0.35) & (y < 0.55) & ((x.square() + z.square()) < 0.85)
        radial = torch.stack([x, torch.zeros_like(x), z], dim=1)
        d[region] = 0.55 * max_d * radial[region]
        description = "central body proxy expands radially in x/z"
    elif name == "ear_expansion":
        # Two upper lateral regions provide a deterministic ear-like test area.
        region = foreground & (y > 0.15) & (y < 0.95) & (x.abs() > 0.20)
        outward = torch.stack([x.sign(), 0.25 * (y - 0.55), 0.15 * z], dim=1)
        local = torch.stack([x, y - 0.55, z], dim=1)
        d[region] = max_d * (0.65 * outward[region] + 0.20 * local[region])
        description = "upper lateral regions expand outward with mild local affine-like motion"
    elif name == "trunk_bending":
        # A narrow front/central vertical corridor is split into y segments.
        region = foreground & (x.abs() < 0.32) & (y > -0.55) & (y < 0.60) & (z < -0.10)
        segment = ((y + 0.55) / 1.15).clamp(0, 0.999)
        bend = (segment - 0.5) * 2.0
        d[region, 0] = max_d * 0.75 * bend[region]
        d[region, 2] = max_d * 0.20 * bend[region].square()
        description = "central front corridor bends progressively across vertical segments"
    else:
        raise ValueError("Unknown deformation: {}".format(name))

    d[~foreground] = 0
    return d, region, description


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    xyz, ply_path = load_xyz(args.model_path, args.load_iteration)
    foreground = torch.load(args.foreground_mask_path, map_location="cpu").bool().flatten()
    if foreground.numel() != xyz.shape[0]:
        raise ValueError("foreground mask length does not match Gaussian count")
    manifest = {"source_ply": os.path.abspath(ply_path), "num_gaussians": int(xyz.shape[0]), "deformations": {}}
    for name in args.deformations:
        delta, region, description = make_deformation(xyz, foreground, name, args.max_d_xyz)
        path = os.path.join(args.output_dir, "synthetic_delta_{}.pt".format(name))
        payload = {
            "d_xyz": delta,
            "d_scaling": torch.zeros_like(delta),
            "d_rotation": torch.zeros((xyz.shape[0], 4)),
            "source_xyz": xyz,
            "foreground_mask": foreground,
            "synthetic_region_mask": region,
            "metadata": {
                "benchmark": "synthetic_known_delta",
                "deformation": name,
                "description": description,
                "max_d_xyz": args.max_d_xyz,
                "d_scaling_disabled": True,
            },
        }
        torch.save(payload, path)
        correspondence_path = os.path.join(args.output_dir, "synthetic_correspondence_{}.pt".format(name))
        torch.save({
            "source_xyz": xyz,
            "target_xyz": xyz + delta,
            "confidence": (foreground & region).float(),
            "foreground_mask": foreground,
            "metadata": {"benchmark": "synthetic_known_delta", "deformation": name},
        }, correspondence_path)
        manifest["deformations"][name] = {
            "delta_path": os.path.abspath(path),
            "active_gaussians": int(region.sum()),
            "mean_norm": float(delta.norm(dim=-1).mean()),
            "foreground_energy_percent": 100.0,
            "background_energy_percent": 0.0,
            "d_scaling_exact_zero": True,
            "description": description,
            "correspondence_path": os.path.abspath(correspondence_path),
        }
        print(name, "active:", int(region.sum()), "mean_norm:", float(delta.norm(dim=-1).mean()))
    with open(os.path.join(args.output_dir, "synthetic_benchmark_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()
