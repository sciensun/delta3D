#!/usr/bin/env python3
"""Evaluate a lifted correspondence payload before it becomes Stage 1 supervision."""
import argparse
import json
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--correspondence_path", required=True)
    p.add_argument("--foreground_mask_path", default=None)
    p.add_argument("--output_path", required=True)
    a = p.parse_args()
    x = torch.load(a.correspondence_path, map_location="cpu")
    source = x["source_xyz"].float(); target = x.get("target_xyz")
    if target is None:
        raise ValueError("correspondence payload needs target_xyz")
    target = target.float(); n = source.shape[0]
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten() if a.foreground_mask_path else x.get("foreground_mask", torch.ones(n).bool())
    conf = x.get("confidence", torch.ones(n)).float().flatten()
    displacement = target - source
    candidates = x.get("candidate_target_xyz")
    if candidates is not None:
        candidates = candidates.float()
        valid = torch.isfinite(candidates).all(-1)
        support = valid.sum(0)
        mean = torch.nanmean(torch.where(valid[..., None], candidates, torch.nan), dim=0)
        residual = ((candidates - mean[None]).square().sum(-1).where(valid, torch.zeros_like(support, dtype=torch.float))).sum(0) / support.clamp_min(1)
        directional_variance = residual
    else:
        support = (conf > 0).long()
        directional_variance = torch.zeros(n)
    high = fg & (conf >= 0.5) & (support >= 2)
    report = {
        "num_gaussians": n, "foreground_gaussians": int(fg.sum()),
        "foreground_centroid": source[fg].mean(0).tolist(),
        "target_centroid": target[fg].mean(0).tolist(),
        "mean_displacement_norm": float(displacement[fg].norm(dim=-1).mean()),
        "high_confidence_gaussians": int(high.sum()),
        "high_confidence_fraction_foreground": float(high.sum() / fg.sum().clamp_min(1)),
        "multi_view_supported_gaussians": int((fg & (support >= 2)).sum()),
        "multi_view_fraction_foreground": float((fg & (support >= 2)).sum() / fg.sum().clamp_min(1)),
        "mean_directional_variance": float(directional_variance[fg].mean()),
        "background_displacement_energy": float(displacement[~fg].square().sum()),
        "d_scaling_disabled": True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
