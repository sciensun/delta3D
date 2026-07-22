#!/usr/bin/env python3
"""Post-recovery evaluator for CPU observed_2d reconstruction."""
import argparse
import json
import os
import torch
import torch.nn.functional as F

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recovered_path", required=True)
    p.add_argument("--ground_truth_path", required=True)
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--output_path", required=True)
    a = p.parse_args()
    pred = torch.load(a.recovered_path, map_location="cpu")["d_xyz"].float()
    gt_payload = torch.load(a.ground_truth_path, map_location="cpu")
    gt = gt_payload["d_xyz"].float()
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    active = gt_payload.get("synthetic_region_mask", fg).bool().flatten()
    eps = 1e-8
    def metrics(mask):
        if not mask.any(): return {"count": 0}
        x, y = pred[mask], gt[mask]
        return {"count": int(mask.sum()),
                "cosine": float(F.cosine_similarity(x.flatten()[None], y.flatten()[None]).item()),
                "energy_ratio": float((x.square().sum() / (y.square().sum() + eps))),
                "explained_variance": float(1 - (x-y).square().sum() / (y-y.mean(0)).square().sum().clamp_min(eps)),
                "mean_norm": float(x.norm(dim=-1).mean()), "median_norm": float(x.norm(dim=-1).median()),
                "p90_norm": float(torch.quantile(x.norm(dim=-1), .9))}
    report = {"global": metrics(torch.ones(len(gt), dtype=torch.bool)),
              "foreground": metrics(fg), "active": metrics(active),
              "background_energy": float(pred[~fg].square().sum()),
              "d_scaling_max": 0.0}
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    json.dump(report, open(a.output_path, "w"), indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
