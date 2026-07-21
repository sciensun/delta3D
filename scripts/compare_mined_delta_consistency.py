#!/usr/bin/env python3
"""Compare independently mined xyz-only deltas on foreground Gaussians."""
import argparse
import json
import os
import warnings

import numpy as np
import torch


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--delta", nargs="+", required=True, help="NAME=PATH entries")
    p.add_argument("--source_xyz_path", default=None)
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--part_labels_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knn", type=int, default=8)
    return p.parse_args()


def load(path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        delta = x["d_xyz"].float()
        xyz = x.get("source_xyz")
        conf = x.get("foreground_support", x.get("confidence", x.get("correspondence_confidence")))
        if conf is None and isinstance(x.get("metadata"), dict) and x["metadata"].get("correspondence_path"):
            try:
                corr = torch.load(x["metadata"]["correspondence_path"], map_location="cpu")
                conf = corr.get("confidence")
            except Exception:
                pass
        fg = x.get("foreground_mask")
    else:
        delta, xyz, conf, fg = x.float(), None, None, None
    return delta, xyz, conf, fg


def cosine_rows(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)


def rank_corr(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a.numpy(), b.numpy()).statistic)
    except Exception:
        ar = torch.argsort(torch.argsort(a)).float()
        br = torch.argsort(torch.argsort(b)).float()
        return float(cosine_rows(ar[:, None] - ar.mean(), br[:, None] - br.mean()).mean())


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8))


def knn_edges(xyz, mask, k):
    idx = torch.where(mask)[0]
    if idx.numel() < 2:
        return None
    if idx.numel() > 4096:
        # Keep diagnostics bounded; pairwise consistency still uses every Gaussian.
        idx = idx[torch.linspace(0, idx.numel() - 1, 4096).long()]
    d = torch.cdist(xyz[idx], xyz[idx])
    d.fill_diagonal_(float("inf"))
    nn = d.topk(min(k, idx.numel() - 1), largest=False).indices
    return idx[:, None].expand_as(nn), idx[nn]


def smoothness(xyz, delta, mask, k):
    e = knn_edges(xyz, mask, k)
    if e is None:
        return 0.0
    a, b = e
    return float((delta[a] - delta[b]).square().sum(dim=-1).mean())


def parse_entries(values):
    out = {}
    for value in values:
        name, path = value.split("=", 1)
        out[name] = path
    return out


def write_color_ply(path, xyz, colors):
    from plyfile import PlyData, PlyElement
    xyz = xyz.numpy().astype(np.float32)
    colors = colors.numpy().clip(0, 255).astype(np.uint8)
    vertices = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(path)


def main():
    a = args()
    os.makedirs(a.output_dir, exist_ok=True)
    entries = parse_entries(a.delta)
    loaded = {name: load(path) for name, path in entries.items()}
    n = next(iter(loaded.values()))[0].shape[0]
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    xyz = next((v[1] for v in loaded.values() if v[1] is not None), None)
    if a.source_xyz_path:
        xyz_payload = torch.load(a.source_xyz_path, map_location="cpu")
        xyz = (xyz_payload["source_xyz"] if isinstance(xyz_payload, dict) else xyz_payload).float()
    if xyz is None:
        raise RuntimeError("source xyz is required in a delta payload or --source_xyz_path")
    labels = torch.load(a.part_labels_path, map_location="cpu").long().flatten() if a.part_labels_path else None
    deltas = {k: v[0] for k, v in loaded.items()}
    names = list(deltas)
    confidences = {k: (v[2].float().flatten() if v[2] is not None else torch.ones(n)) for k, v in loaded.items()}
    has_confidence = any(v[2] is not None for v in loaded.values())
    high_conf = fg & (torch.stack([confidences[k] for k in names]).mean(0) >= 0.5) if has_confidence else torch.zeros_like(fg)
    if any(v.shape != (n, 3) for v in deltas.values()) or len(fg) != n:
        raise ValueError("all deltas and the foreground mask must have matching [N,3]/[N] shapes")
    pair_metrics = {}
    agreement = torch.zeros(n)
    pair_count = torch.zeros(n)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            x, y = deltas[left][fg], deltas[right][fg]
            c = cosine_rows(x, y)
            nx, ny = x.norm(dim=-1), y.norm(dim=-1)
            w = (nx * ny).clamp_min(1e-8)
            weighted = float((c * w).sum() / w.sum())
            cw = (confidences[left][fg] * confidences[right][fg]).clamp_min(0.0) * w
            confidence_weighted = float((c * cw).sum() / cw.sum().clamp_min(1e-8))
            hc = high_conf[fg]
            pair = {
                "left": left, "right": right,
                "global_cosine": weighted,
                "confidence_weighted_cosine": confidence_weighted,
                "high_confidence_cosine": float(c[hc].mean()) if hc.any() else None,
                "high_confidence_count": int(hc.sum()),
                "mean_per_gaussian_cosine": float(c.mean()),
                "median_per_gaussian_cosine": float(c.quantile(.5)),
                "p25_per_gaussian_cosine": float(c.quantile(.25)),
                "p75_per_gaussian_cosine": float(c.quantile(.75)),
                "magnitude_pearson": pearson(nx, ny),
                "magnitude_spearman": rank_corr(nx, ny),
                "relative_norm_difference": float((nx - ny).norm() / ny.norm().clamp_min(1e-8)),
                "direction_agreement_fraction": float((c > .5).float().mean()),
                "graph_smoothness_left": smoothness(xyz, deltas[left], fg, a.knn),
                "graph_smoothness_right": smoothness(xyz, deltas[right], fg, a.knn),
            }
            pair_metrics["{}__{}".format(left, right)] = pair
            idx = torch.where(fg)[0]
            agreement[idx] += c
            pair_count[idx] += 1
    per = agreement / pair_count.clamp_min(1)
    colors = torch.zeros((n, 3), dtype=torch.uint8)
    # Green = agreeing motion, red = uncertain, blue = directional conflict.
    colors[fg, 0] = ((1 - per[fg].clamp(0, 1)) * 255).byte()
    colors[fg, 1] = (per[fg].clamp(0, 1) * 255).byte()
    colors[fg, 2] = ((-per[fg]).clamp(0, 1) * 255).byte()
    part_metrics = {}
    if labels is not None:
        for part_id in sorted(int(x) for x in labels[fg].unique().tolist() if int(x) >= 0):
            pm = fg & (labels == part_id)
            part_metrics[str(part_id)] = {
                "count": int(pm.sum()),
                "mean_agreement": float(per[pm].mean()),
                "median_agreement": float(per[pm].quantile(.5)),
                "conflict_fraction": float((per[pm] < 0).float().mean()),
                "high_agreement_fraction": float((per[pm] > .5).float().mean()),
            }
    torch.save({"mean_pairwise_cosine": per, "foreground_mask": fg, "source_xyz": xyz},
               os.path.join(a.output_dir, "per_gaussian_consistency.pt"))
    write_color_ply(os.path.join(a.output_dir, "consistency_color.ply"), xyz, colors)
    report = {"deltas": {k: os.path.abspath(v) for k, v in entries.items()},
              "num_gaussians": n, "foreground_gaussians": int(fg.sum()),
              "pair_metrics": pair_metrics,
              "mean_pairwise_cosine": float(per[fg].mean()) if fg.any() else 0.0,
              "median_pairwise_cosine": float(per[fg].quantile(.5)) if fg.any() else 0.0,
              "high_confidence_fraction": float((per[high_conf] > .5).float().mean()) if high_conf.any() else None,
              "direction_conflict_fraction": float((per[fg] < 0).float().mean()) if fg.any() else 0.0,
              "part_metrics": part_metrics,
              "top_agreement_parts": sorted(part_metrics, key=lambda k: part_metrics[k]["mean_agreement"], reverse=True)[:5],
              "top_conflict_parts": sorted(part_metrics, key=lambda k: part_metrics[k]["conflict_fraction"], reverse=True)[:5],
              "decision": "strong" if pair_metrics and min(x["global_cosine"] for x in pair_metrics.values()) >= .7
                          else "mixed" if pair_metrics and min(x["global_cosine"] for x in pair_metrics.values()) >= .5
                          else "weak_or_unavailable"}
    with open(os.path.join(a.output_dir, "consistency_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
