#!/usr/bin/env python3
"""Evaluate an observed_2d-only delta against hidden synthetic ground truth.

The ground-truth delta is read only by this evaluator, after optimization.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def project(points, camera):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    clip = torch.cat([points, ones], dim=1) @ camera.full_proj_transform
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)
    return torch.stack([(ndc[:, 0] * .5 + .5) * float(camera.image_width),
                        (1 - (ndc[:, 1] * .5 + .5)) * float(camera.image_height)], dim=1)


def rank_corr(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a.numpy(), b.numpy()).statistic)
    except Exception:
        ar = torch.argsort(torch.argsort(a)).float()
        br = torch.argsort(torch.argsort(b)).float()
        ar -= ar.mean(); br -= br.mean()
        return float((ar * br).sum() / (ar.norm() * br.norm()).clamp_min(1e-8))


def knn_distortion(source, moved, mask, k=8, limit=4096):
    ids = torch.where(mask)[0]
    if ids.numel() < 2:
        return {"median": None, "p05": None, "p95": None}
    if ids.numel() > limit:
        ids = ids[torch.linspace(0, ids.numel() - 1, limit).long()]
    before = torch.cdist(source[ids], source[ids])
    after = torch.cdist(moved[ids], moved[ids])
    before.fill_diagonal_(float("inf")); after.fill_diagonal_(float("inf"))
    neighbors = before.topk(min(k, len(ids) - 1), largest=False).indices
    ratio = after.gather(1, neighbors) / before.gather(1, neighbors).clamp_min(1e-8)
    return {"median": float(ratio.median()), "p05": float(ratio.quantile(.05)), "p95": float(ratio.quantile(.95))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_path", required=True)
    p.add_argument("--recovered_path", required=True)
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--reference_bundle", required=True)
    p.add_argument("--source_path", required=True)
    p.add_argument("--novel_source_path", default=None,
                   help="Optional larger camera set used only for post-hoc novel-view projection evaluation.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_view_indices", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--output_path", required=True)
    a = p.parse_args()
    gt = torch.load(a.ground_truth_path, map_location="cpu")
    rec = torch.load(a.recovered_path, map_location="cpu")
    source = gt["source_xyz"].float(); target = gt["d_xyz"].float(); pred = rec["d_xyz"].float()
    fg = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten()
    active = gt.get("synthetic_region_mask", fg).bool().flatten()
    row = F.cosine_similarity(pred[fg], target[fg], dim=-1)
    active_row = F.cosine_similarity(pred[active], target[active], dim=-1)
    diff = pred[fg] - target[fg]
    centered = target[fg] - target[fg].mean(0, keepdim=True)
    gt_energy = target[fg].square().sum().clamp_min(1e-12)
    report = {
        "recovered_path": os.path.abspath(a.recovered_path),
        "global_cosine": float(F.cosine_similarity(pred[fg].flatten()[None], target[fg].flatten()[None]).item()),
        "active_region_cosine": float(F.cosine_similarity(pred[active].flatten()[None], target[active].flatten()[None]).item()) if active.any() else None,
        "mean_per_gaussian_cosine": float(row.mean()),
        "median_per_gaussian_cosine": float(row.quantile(.5)),
        "energy_ratio": float(pred[fg].square().sum() / gt_energy),
        "explained_variance": float(1 - diff.square().sum() / centered.square().sum().clamp_min(1e-12)),
        "magnitude_pearson": float(torch.corrcoef(torch.stack([pred[fg].norm(dim=-1), target[fg].norm(dim=-1)]))[0, 1]),
        "magnitude_spearman": rank_corr(pred[fg].norm(dim=-1), target[fg].norm(dim=-1)),
        "direction_conflict_fraction": float((row < 0).float().mean()),
        "background_energy": float(pred[~fg].square().sum()),
        "d_scaling_max": float(rec.get("d_scaling", torch.zeros_like(pred)).abs().max()),
        "knn_edge_distortion": knn_distortion(source, source + pred, fg),
        "train_view_indices": [int(x) for x in a.train_view_indices.split(",") if x.strip()],
    }

    # This is a held-out projection/rendering proxy: the clean observation bundle
    # supplies hidden target pixels, while the recovered delta is projected only
    # after optimization. No target xyz is used in this calculation.
    bundle = torch.load(a.reference_bundle, map_location="cpu")
    target_xy = bundle["target_xy"].float()
    visibility = bundle.get("visibility_2d", bundle.get("visibility"))
    visibility = torch.ones(target_xy.shape[:2], dtype=torch.bool) if visibility is None else visibility.bool()
    parser = argparse.ArgumentParser(add_help=False); lp = ModelParams(parser)
    args = parser.parse_args(["-s", a.source_path, "--model_path", a.model_path])
    safe_state(True)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(lp.extract(args), gaussians, load_iteration=a.load_iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    errors = []
    train_indices = set(report["train_view_indices"])
    for view_index in range(min(len(cameras), target_xy.shape[0])):
        if view_index in train_indices:
            continue
        predicted_xy = project((source + pred).cuda(), cameras[view_index]).cpu()
        valid = visibility[view_index] & fg
        if valid.any():
            errors.append(float((predicted_xy[valid] - target_xy[view_index, valid]).norm(dim=-1).mean()))
    if a.novel_source_path:
        novel_parser = argparse.ArgumentParser(add_help=False); novel_lp = ModelParams(novel_parser)
        novel_cli = novel_parser.parse_args(["-s", a.novel_source_path, "--model_path", a.model_path])
        novel_scene = Scene(novel_lp.extract(novel_cli), gaussians, load_iteration=a.load_iteration, shuffle=False)
        novel_errors = []
        for camera in novel_scene.getTrainCameras():
            truth_xy = project((source + target).cuda(), camera).cpu()
            predicted_xy = project((source + pred).cuda(), camera).cpu()
            valid = fg
            if valid.any():
                novel_errors.append(float((predicted_xy[valid] - truth_xy[valid]).norm(dim=-1).mean()))
        errors = novel_errors
    report["novel_view_projection_rmse_px"] = float(sum(errors) / len(errors)) if errors else None
    report["novel_view_count"] = len(errors)
    report["novel_view_quality"] = "coherent_proxy" if errors and report["novel_view_projection_rmse_px"] < 5.0 else "not_established"
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
