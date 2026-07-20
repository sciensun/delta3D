#!/usr/bin/env python3
"""Build a confidence-weighted consensus from independent xyz-only deltas."""
import argparse
import json
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta", nargs="+", required=True, help="NAME=PATH entries")
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--output_path", required=True)
    a = p.parse_args()
    entries = dict(x.split("=", 1) for x in a.delta)
    payloads = [torch.load(path, map_location="cpu") for path in entries.values()]
    deltas = torch.stack([x["d_xyz"].float() for x in payloads])
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    norms = deltas.norm(dim=-1)
    unit = deltas / norms.unsqueeze(-1).clamp_min(1e-8)
    center = unit.mean(dim=0)
    agreement = torch.nn.functional.cosine_similarity(unit, center[None], dim=-1).mean(dim=0)
    weights = agreement.clamp_min(0.0) * fg.float()
    weights = weights / weights.sum().clamp_min(1e-8)
    consensus = deltas.mean(dim=0)
    # Suppress directions that disagree across splits; preserve zero background exactly.
    consensus = consensus * agreement.clamp_min(0.0).unsqueeze(-1) * fg[:, None]
    disagreement = (1.0 - agreement).clamp(0, 2)
    source_xyz = payloads[0].get("source_xyz") if isinstance(payloads[0], dict) else None
    out = {"d_xyz": consensus, "d_scaling": torch.zeros_like(consensus),
           "foreground_mask": fg, "consensus_confidence": agreement,
           "disagreement": disagreement, "source_split_paths": entries,
           "metadata": {"method": "direction-gated mean", "num_splits": len(entries)}}
    if source_xyz is not None:
        out["source_xyz"] = source_xyz.float()
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    torch.save(out, a.output_path)
    with open(os.path.splitext(a.output_path)[0] + "_report.json", "w", encoding="utf-8") as f:
        json.dump({"num_splits": len(entries), "foreground_gaussians": int(fg.sum()),
                   "mean_confidence": float(agreement[fg].mean()) if fg.any() else 0.0,
                   "mean_disagreement": float(disagreement[fg].mean()) if fg.any() else 0.0}, f, indent=2)
    print("saved:", a.output_path)
    print("mean foreground confidence:", float(agreement[fg].mean()) if fg.any() else 0.0)


if __name__ == "__main__":
    main()
