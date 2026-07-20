#!/usr/bin/env python3
"""Fit a compact part dictionary to a successful foreground-only delta."""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model_path", required=True)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--foreground_mask_path", required=True)
    parser.add_argument("--part_labels_path", required=True)
    parser.add_argument("--confidence_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--representation", choices=("translation", "affine"), required=True)
    parser.add_argument("--soft_assignment", action="store_true")
    parser.add_argument("--soft_topk", type=int, default=3)
    parser.add_argument("--soft_sigma", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_affine", type=float, default=1e-3)
    parser.add_argument("--lambda_boundary", type=float, default=1e-2)
    parser.add_argument("--lambda_energy", type=float, default=0.1)
    parser.add_argument("--huber_delta", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1e-5)
    return parser.parse_args()


def load_xyz(model_path, iteration):
    from plyfile import PlyData

    path = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    xyz = torch.from_numpy(
        __import__("numpy").stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).copy()
    ).float()
    return xyz, path


def load_vector(path, dtype=torch.float32):
    return torch.load(path, map_location="cpu").to(dtype=dtype).flatten()


def build_weights(xyz, labels, foreground, soft, topk, sigma):
    num_parts = int(labels[foreground].max().item()) + 1
    centers = []
    for part_id in range(num_parts):
        mask = foreground & (labels == part_id)
        if not mask.any():
            raise ValueError("Part {} has no foreground Gaussians.".format(part_id))
        centers.append(xyz[mask].mean(dim=0))
    centers = torch.stack(centers)
    weights = torch.zeros((xyz.shape[0], num_parts), dtype=xyz.dtype)
    if soft:
        distances = torch.cdist(xyz[foreground], centers)
        k = min(max(1, topk), num_parts)
        values, indices = torch.topk(distances, k=k, largest=False, dim=1)
        scale = values[:, -1:].clamp_min(float(sigma) * 1e-3)
        scores = torch.exp(-((values / scale) ** 2))
        scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights[foreground] = torch.zeros_like(distances).scatter(1, indices, scores)
    else:
        assignments = torch.cdist(xyz[foreground], centers).argmin(dim=1)
        weights[foreground, assignments] = 1.0
    return weights, centers


def solve_ridge(design, target, confidence, ridge):
    design = design.float()
    target = target.float()
    confidence = confidence.float().clamp_min(0.0)
    weighted_design = design * confidence.sqrt()[:, None]
    weighted_target = target * confidence.sqrt()[:, None]
    gram = weighted_design.T @ weighted_design
    gram = gram + torch.eye(gram.shape[0]) * ridge
    rhs = weighted_design.T @ weighted_target
    return torch.linalg.solve(gram, rhs)


def predict(xyz, weights, centers, coefficients, representation):
    num_parts = centers.shape[0]
    if representation == "translation":
        return weights @ coefficients[:, :3]
    translations = coefficients[:, :3]
    affine = coefficients[:, 3:].reshape(num_parts, 3, 3).transpose(1, 2)
    local = xyz[:, None, :] - centers[None, :, :]
    affine_delta = torch.einsum("nki,kji->nkj", local, affine)
    return (weights[:, :, None] * (translations[None] + affine_delta)).sum(dim=1)


def fit_coefficients(xyz, target, weights, centers, confidence, representation, ridge):
    foreground = confidence > 0
    x = xyz[foreground]
    w = weights[foreground]
    y = target[foreground]
    c = confidence[foreground]
    num_parts = centers.shape[0]
    if representation == "translation":
        coefficients = solve_ridge(w, y, c, ridge)
        return torch.cat([coefficients, torch.zeros((num_parts, 9))], dim=1)
    local = x[:, None, :] - centers[None, :, :]
    design = torch.cat([w[:, :, None], w[:, :, None] * local], dim=2).reshape(x.shape[0], num_parts * 4)
    solved = solve_ridge(design, y, c, ridge).reshape(num_parts, 4, 3)
    translations = solved[:, 0, :]
    affine = solved[:, 1:, :].reshape(num_parts, 9)
    return torch.cat([translations, affine], dim=1)


def metrics(pred, target, foreground, labels, confidence, representation, coefficients):
    mask = foreground.bool()
    diff = pred[mask] - target[mask]
    conf = confidence[mask].clamp_min(0.0)
    denom = conf.sum().clamp_min(1e-8)
    weighted_mse = (conf[:, None] * diff.square()).sum() / (denom * 3.0)
    target_rms = target[mask].square().mean().sqrt().clamp_min(1e-8)
    pred_flat = pred[mask].reshape(-1)
    target_flat = target[mask].reshape(-1)
    centered = target_flat - target_flat.mean()
    energy_target = target[mask].square().sum().clamp_min(1e-12)
    energy_pred = pred[mask].square().sum()
    stats = {
        "weighted_rmse": float(weighted_mse.sqrt()),
        "normalized_rmse": float(weighted_mse.sqrt() / target_rms),
        "global_cosine": float(torch.nn.functional.cosine_similarity(pred_flat[None], target_flat[None]).item()),
        "explained_variance": float(1.0 - diff.square().sum() / centered.square().sum().clamp_min(1e-12)),
        "energy_preservation_ratio": float(energy_pred / energy_target),
        "foreground_delta_energy_percent": 100.0,
        "background_delta_energy_percent": 0.0,
        "target_rms": float(target_rms),
        "pred_rms": float(pred[mask].square().mean().sqrt()),
        "d_scaling_exact_zero": True,
        "representation": representation,
    }
    pred_norm = pred[mask].norm(dim=-1)
    target_norm = target[mask].norm(dim=-1)
    stats.update({
        "pred_mean_norm": float(pred_norm.mean()),
        "pred_median_norm": float(pred_norm.quantile(0.5)),
        "pred_p90_norm": float(pred_norm.quantile(0.9)),
        "pred_p95_norm": float(pred_norm.quantile(0.95)),
        "pred_max_norm": float(pred_norm.max()),
        "target_p95_norm": float(target_norm.quantile(0.95)),
        "p95_norm_ratio": float(pred_norm.quantile(0.95) / target_norm.quantile(0.95).clamp_min(1e-8)),
    })
    part_stats = []
    for part_id in sorted(int(x) for x in labels.unique().tolist() if int(x) >= 0):
        part_mask = mask & (labels == part_id)
        if not part_mask.any():
            continue
        part_stats.append({
            "part": part_id,
            "count": int(part_mask.sum()),
            "target_mean_norm": float(target[part_mask].norm(dim=-1).mean()),
            "pred_mean_norm": float(pred[part_mask].norm(dim=-1).mean()),
            "target_energy": float(target[part_mask].square().sum() / energy_target),
            "pred_energy": float(pred[part_mask].square().sum() / energy_target),
        })
    stats["part_stats"] = sorted(part_stats, key=lambda item: item["pred_energy"], reverse=True)
    return stats


def build_boundary_pairs(xyz, labels, foreground, max_pairs=4096):
    indices = torch.where(foreground)[0]
    if indices.numel() < 2:
        return None
    sample_count = min(4096, indices.numel())
    sample = indices[torch.linspace(0, indices.numel() - 1, sample_count).long()]
    distances = torch.cdist(xyz[sample], xyz[sample])
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=1)
    boundary = labels[sample] != labels[sample[nearest]]
    left = sample[boundary]
    right = sample[nearest[boundary]]
    if left.numel() > max_pairs:
        left = left[:max_pairs]
        right = right[:max_pairs]
    return left, right


def refine_coefficients(coefficients, xyz, target, weights, centers, labels, foreground, confidence, args):
    parameter = torch.nn.Parameter(coefficients.clone())
    optimizer = torch.optim.Adam([parameter], lr=args.lr)
    edges = build_boundary_pairs(xyz, labels, foreground)
    target_energy = target[foreground].square().sum().clamp_min(1e-12)
    for _ in range(max(0, args.iterations)):
        prediction = predict(xyz, weights, centers, parameter, args.representation)
        fit = F.huber_loss(prediction[foreground], target[foreground], reduction="none", delta=args.huber_delta).mean(dim=-1)
        fit = (fit * confidence[foreground]).sum() / confidence[foreground].sum().clamp_min(1e-8)
        loss = fit
        if args.representation == "affine":
            loss = loss + args.lambda_affine * parameter[:, 3:].square().mean()
        if edges is not None and args.lambda_boundary > 0:
            left, right = edges
            loss = loss + args.lambda_boundary * (prediction[left] - prediction[right]).square().mean()
        predicted_energy = prediction[foreground].square().sum()
        loss = loss + args.lambda_energy * (predicted_energy / target_energy - 1.0).square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return parameter.detach()


def main():
    args = parse_args()
    payload = torch.load(args.mined_delta_path, map_location="cpu")
    target = payload["d_xyz"].float()
    foreground = load_vector(args.foreground_mask_path, dtype=torch.bool)
    labels = load_vector(args.part_labels_path, dtype=torch.long)
    if target.shape[0] != foreground.shape[0] or target.shape[0] != labels.shape[0]:
        raise ValueError("delta, foreground mask, and part labels must have the same length")
    xyz, ply_path = load_xyz(args.source_model_path, args.load_iteration)
    confidence = torch.ones(target.shape[0])
    if args.confidence_path:
        confidence = load_vector(args.confidence_path)
    confidence = confidence * foreground.float()
    weights, centers = build_weights(xyz, labels, foreground, args.soft_assignment, args.soft_topk, args.soft_sigma)
    coefficients = fit_coefficients(xyz, target, weights, centers, confidence, args.representation, args.ridge)
    coefficients = refine_coefficients(
        coefficients, xyz, target, weights, centers, labels, foreground, confidence, args
    )
    pred = predict(xyz, weights, centers, coefficients, args.representation)
    report = metrics(pred, target, foreground, labels, confidence, args.representation, coefficients)
    report.update({
        "source_model_path": os.path.abspath(args.source_model_path),
        "source_ply_path": os.path.abspath(ply_path),
        "mined_delta_path": os.path.abspath(args.mined_delta_path),
        "foreground_mask_path": os.path.abspath(args.foreground_mask_path),
        "part_labels_path": os.path.abspath(args.part_labels_path),
        "soft_assignment": args.soft_assignment,
        "soft_topk": args.soft_topk,
        "num_parts": int(centers.shape[0]),
        "iterations": args.iterations,
        "lr": args.lr,
        "lambda_affine": args.lambda_affine,
        "lambda_boundary": args.lambda_boundary,
        "lambda_energy": args.lambda_energy,
        "huber_delta": args.huber_delta,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    delta_payload = {
        "d_xyz": pred.detach(),
        "d_scaling": torch.zeros_like(pred),
        "d_rotation": torch.zeros((pred.shape[0], 4)),
        "source_xyz": xyz,
        "foreground_mask": foreground,
        "part_labels": labels,
        "part_centers": centers,
        "part_weights": weights,
        "part_coefficients": coefficients,
        "metrics": report,
        "metadata": {"representation": args.representation, "d_scaling_disabled": True},
    }
    torch.save(delta_payload, output_path)
    torch.save({
        "representation": args.representation,
        "part_centers": centers,
        "part_translations": coefficients[:, :3],
        "part_affine": coefficients[:, 3:].reshape(centers.shape[0], 3, 3).transpose(1, 2),
        "metrics": report,
    }, os.path.join(output_dir, "part_dictionary.pt"))
    torch.save({
        "part_weights": weights,
        "part_labels": labels,
        "foreground_mask": foreground,
        "soft_assignment": args.soft_assignment,
        "soft_topk": args.soft_topk,
    }, os.path.join(output_dir, "part_coefficients.pt"))
    with open(os.path.join(output_dir, "partfit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("representation:", args.representation)
    print("soft assignment:", args.soft_assignment)
    print("weighted RMSE: {:.8f}".format(report["weighted_rmse"]))
    print("normalized RMSE: {:.8f}".format(report["normalized_rmse"]))
    print("global cosine: {:.8f}".format(report["global_cosine"]))
    print("explained variance: {:.8f}".format(report["explained_variance"]))
    print("energy preservation ratio: {:.8f}".format(report["energy_preservation_ratio"]))
    print("saved delta:", output_path)
    print("saved dictionary:", os.path.join(output_dir, "part_dictionary.pt"))
    print("saved coefficients:", os.path.join(output_dir, "part_coefficients.pt"))


if __name__ == "__main__":
    main()
