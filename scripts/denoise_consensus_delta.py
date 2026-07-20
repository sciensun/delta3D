#!/usr/bin/env python3
"""Graph-regularized CPU denoising of a consensus xyz-only delta."""
import argparse
import json
import os
import torch
import torch.nn.functional as F


def edges(xyz, mask, k):
    ids = torch.where(mask)[0]
    d = torch.cdist(xyz[ids], xyz[ids]); d.fill_diagonal_(float("inf"))
    nn = d.topk(min(k, len(ids) - 1), largest=False).indices
    return ids[:, None].expand_as(nn).flatten(), ids[nn].flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--consensus_path", required=True); p.add_argument("--source_xyz_path", required=True)
    p.add_argument("--foreground_mask_path", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    p.add_argument("--knn", type=int, default=8); p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-2)
    a = p.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    c = torch.load(a.consensus_path, map_location="cpu"); target = c["d_xyz"].float()
    xyz_payload = torch.load(a.source_xyz_path, map_location="cpu")
    xyz = (xyz_payload["source_xyz"] if isinstance(xyz_payload, dict) else xyz_payload).float()
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    left, right = edges(xyz, fg, a.knn); rows = []
    target_e = target[fg].square().sum().clamp_min(1e-12)
    for lam in a.lambdas:
        d = torch.nn.Parameter(target.clone()); opt = torch.optim.Adam([d], lr=a.lr)
        for _ in range(a.iterations):
            fit = F.huber_loss(d[fg], target[fg], delta=.01)
            smooth = (d[left] - d[right]).square().sum(dim=-1).mean()
            loss = fit + lam * smooth
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): d[~fg].zero_()
        with torch.no_grad():
            pred = d.detach(); diff = pred[fg] - target[fg]
            cosine = F.cosine_similarity(pred[fg].flatten()[None], target[fg].flatten()[None]).item()
            energy = float(pred[fg].square().sum() / target_e)
            ev = float(1 - diff.square().sum() / (target[fg] - target[fg].mean()).square().sum().clamp_min(1e-12))
            smooth_v = float((pred[left] - pred[right]).square().sum(dim=-1).mean())
            path = os.path.join(a.output_dir, "consensus_denoised_lambda_{:.0e}.pt".format(lam))
            torch.save({"d_xyz": pred, "d_scaling": torch.zeros_like(pred), "foreground_mask": fg,
                        "consensus_path": os.path.abspath(a.consensus_path), "lambda_smooth": lam}, path)
            rows.append({"lambda_smooth": lam, "cosine_to_consensus": cosine, "energy_ratio": energy,
                         "explained_variance": ev, "graph_smoothness": smooth_v, "path": path})
    with open(os.path.join(a.output_dir, "denoising_metrics.json"), "w", encoding="utf-8") as f: json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
