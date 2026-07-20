#!/usr/bin/env python3
"""Evaluate whether a mined delta is foreground-focused and part-structured."""

import argparse
import json
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--foreground_mask_path", default=None)
    parser.add_argument("--part_labels_path", default=None)
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.mined_delta_path, map_location="cpu")
    d_xyz = payload["d_xyz"].float()
    d_scaling = payload["d_scaling"].float()
    n = d_xyz.shape[0]
    fg = payload.get("foreground_mask")
    if args.foreground_mask_path:
        fg = torch.load(args.foreground_mask_path, map_location="cpu")
    if fg is None:
        fg = torch.ones(n, dtype=torch.bool)
    fg = fg.bool().flatten()
    labels = payload.get("part_labels")
    if args.part_labels_path:
        labels = torch.load(args.part_labels_path, map_location="cpu")
    labels = labels.long().flatten() if labels is not None else None

    xyz_energy = (d_xyz ** 2).sum(dim=-1)
    total_energy = xyz_energy.sum().clamp_min(1e-12)
    fg_energy = xyz_energy[fg].sum()
    bg_energy = xyz_energy[~fg].sum()
    scaling_max = d_scaling.norm(dim=-1).max().item()

    report = {
        "mined_delta_path": os.path.abspath(args.mined_delta_path),
        "num_gaussians": int(n),
        "foreground_gaussians": int(fg.sum()),
        "background_gaussians": int((~fg).sum()),
        "foreground_delta_energy_ratio": float(fg_energy / total_energy),
        "background_delta_energy_ratio": float(bg_energy / total_energy),
        "d_scaling_max_norm": float(scaling_max),
    }

    print("num Gaussians:", report["num_gaussians"])
    print("foreground Gaussians:", report["foreground_gaussians"])
    print("background Gaussians:", report["background_gaussians"])
    print("foreground delta energy ratio: {:.6f}".format(report["foreground_delta_energy_ratio"]))
    print("background delta energy ratio: {:.6f}".format(report["background_delta_energy_ratio"]))
    print("d_scaling max norm: {:.8f}".format(scaling_max))

    if labels is not None:
        part_stats = []
        for part_id in sorted(int(x) for x in labels.unique().tolist() if int(x) >= 0):
            mask = labels == part_id
            norms = d_xyz[mask].norm(dim=-1)
            energy = xyz_energy[mask].sum()
            stat = {
                "part": part_id,
                "count": int(mask.sum()),
                "mean_norm": float(norms.mean()) if norms.numel() else 0.0,
                "p95_norm": float(torch.quantile(norms, torch.tensor(0.95))) if norms.numel() else 0.0,
                "energy_ratio": float(energy / total_energy),
            }
            part_stats.append(stat)
        part_stats = sorted(part_stats, key=lambda x: x["energy_ratio"], reverse=True)
        report["part_stats"] = part_stats
        print("top moving parts:")
        for stat in part_stats[:8]:
            print(
                "  part {part}: count={count} mean={mean_norm:.6f} p95={p95_norm:.6f} energy={energy_ratio:.6f}".format(
                    **stat
                )
            )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
