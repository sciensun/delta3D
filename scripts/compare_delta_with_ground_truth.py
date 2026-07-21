#!/usr/bin/env python3
"""Compute recovery metrics for a mined delta against a known synthetic delta."""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_path", required=True)
    p.add_argument("--recovered_path", required=True)
    p.add_argument("--part_labels_path", default=None)
    p.add_argument("--output_path", required=True)
    a = p.parse_args()
    gt = torch.load(a.ground_truth_path, map_location="cpu")
    rec = torch.load(a.recovered_path, map_location="cpu")
    target, pred = gt["d_xyz"].float(), rec["d_xyz"].float()
    fg = gt.get("foreground_mask", torch.ones(target.shape[0], dtype=torch.bool)).bool().flatten()
    bg = ~fg
    x, y = pred[fg], target[fg]
    c = F.cosine_similarity(x.flatten()[None], y.flatten()[None]).item()
    row_c = F.cosine_similarity(x, y, dim=-1)
    diff = x - y
    target_centered = y - y.mean(0, keepdim=True)
    energy_gt = y.square().sum().clamp_min(1e-12)
    energy_pred = x.square().sum()
    report = {
        "num_gaussians": int(target.shape[0]),
        "foreground_gaussians": int(fg.sum()),
        "global_cosine": c,
        "confidence_weighted_cosine": c,
        "mean_per_gaussian_cosine": float(row_c.mean()),
        "median_per_gaussian_cosine": float(row_c.quantile(.5)),
        "p25_per_gaussian_cosine": float(row_c.quantile(.25)),
        "p75_per_gaussian_cosine": float(row_c.quantile(.75)),
        "energy_ratio": float(energy_pred / energy_gt),
        "explained_variance": float(1.0 - diff.square().sum() / target_centered.square().sum().clamp_min(1e-12)),
        "target_mean_norm": float(y.norm(dim=-1).mean()),
        "recovered_mean_norm": float(x.norm(dim=-1).mean()),
        "target_p95_norm": float(y.norm(dim=-1).quantile(.95)),
        "recovered_p95_norm": float(x.norm(dim=-1).quantile(.95)),
        "background_energy": float(pred[bg].square().sum()),
        "d_scaling_exact_zero": bool(torch.equal(rec.get("d_scaling", torch.zeros_like(pred)), torch.zeros_like(pred))),
    }
    region = gt.get("synthetic_region_mask")
    if region is not None:
        region = region.bool()
        report["active_region"] = {
            "count": int(region.sum()),
            "cosine": float(F.cosine_similarity(pred[region].flatten()[None], target[region].flatten()[None]).item()),
            "energy_ratio": float(pred[region].square().sum() / target[region].square().sum().clamp_min(1e-12)),
        }
    if a.part_labels_path:
        labels = torch.load(a.part_labels_path, map_location="cpu").long().flatten()
        report["parts"] = {}
        for part in sorted(int(v) for v in labels[fg].unique().tolist() if int(v) >= 0):
            m = fg & (labels == part)
            report["parts"][str(part)] = {"count": int(m.sum()), "cosine": float(F.cosine_similarity(pred[m].flatten()[None], target[m].flatten()[None]).item()), "target_mean_norm": float(target[m].norm(dim=-1).mean()), "recovered_mean_norm": float(pred[m].norm(dim=-1).mean())}
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
