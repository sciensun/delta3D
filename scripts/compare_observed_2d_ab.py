#!/usr/bin/env python3
"""Compare two observed_2d mined deltas without exposing target xyz."""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_a", required=True); p.add_argument("--delta_b", required=True)
    p.add_argument("--foreground_mask_path", required=True); p.add_argument("--region_mask_path", default=None); p.add_argument("--output_path", required=True)
    a = p.parse_args()
    x = torch.load(a.delta_a, map_location="cpu")["d_xyz"].float()
    y = torch.load(a.delta_b, map_location="cpu")["d_xyz"].float()
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    if a.region_mask_path:
        region_payload = torch.load(a.region_mask_path, map_location="cpu")
        region = (region_payload.get("synthetic_region_mask", region_payload)
                  if isinstance(region_payload, dict) else region_payload).bool().flatten()
    else:
        region = None
    row = F.cosine_similarity(x[fg], y[fg], dim=-1)
    wx = x[fg].norm(dim=-1); wy = y[fg].norm(dim=-1)
    weight = (wx * wy).clamp_min(1e-8)
    report = {
        "weighted_cosine": float((row * weight).sum() / weight.sum()),
        "median_per_gaussian_cosine": float(row.quantile(.5)),
        "direction_conflict_fraction": float((row < 0).float().mean()),
        "direction_agreement_fraction": float((row > .5).float().mean()),
        "relative_norm_difference": float((wx - wy).norm() / wy.norm().clamp_min(1e-8)),
        "background_energy_a": float(x[~fg].square().sum()),
        "background_energy_b": float(y[~fg].square().sum()),
    }
    if region is not None:
        active = fg & region
        active_row = F.cosine_similarity(x[active], y[active], dim=-1)
        active_weight = (x[active].norm(dim=-1) * y[active].norm(dim=-1)).clamp_min(1e-8)
        report["active_region"] = {
            "count": int(active.sum()),
            "weighted_cosine": float((active_row * active_weight).sum() / active_weight.sum()),
            "median_cosine": float(active_row.quantile(.5)),
            "direction_conflict_fraction": float((active_row < 0).float().mean()),
        }
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
