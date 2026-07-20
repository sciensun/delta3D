#!/usr/bin/env python3
"""Cluster foreground Gaussians into part-like units for Stage 1 experiments."""

import argparse
import json
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--foreground_mask_path", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_parts", type=int, default=16)
    parser.add_argument("--max_kmeans_iter", type=int, default=80)
    return parser.parse_args()


def load_features(model_path, iteration):
    ply_path = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    sh_names = [name for name in names if name.startswith("f_dc_") or name.startswith("f_rest_")]
    sh = np.stack([v[name] for name in sh_names], axis=1).astype(np.float32)
    return ply, xyz, scale, sh, ply_path


def normalize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def kmeans(features, k, max_iter=80, seed=0):
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    centers = features[rng.choice(n, size=k, replace=n < k)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for part_id in range(k):
            mask = labels == part_id
            if mask.any():
                centers[part_id] = features[mask].mean(axis=0)
    return labels, centers


def write_cluster_ply(path, xyz, labels, num_parts):
    rng = np.random.default_rng(7)
    colors = rng.integers(40, 255, size=(num_parts, 3), dtype=np.uint8)
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    fg = labels >= 0
    rgb[fg] = colors[labels[fg]]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    data = np.empty(xyz.shape[0], dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def main():
    args = parse_args()
    out_dir = args.out_dir or args.model_path
    os.makedirs(out_dir, exist_ok=True)
    ply, xyz, scale, sh, ply_path = load_features(args.model_path, args.load_iteration)
    fg_mask = torch.load(args.foreground_mask_path, map_location="cpu").bool().numpy()
    if fg_mask.shape[0] != xyz.shape[0]:
        raise ValueError("foreground mask length does not match Gaussian count.")
    fg_idx = np.where(fg_mask)[0]
    features = np.concatenate([normalize(xyz[fg_idx]), normalize(scale[fg_idx]), normalize(sh[fg_idx]) * 0.25], axis=1)
    fg_labels, _ = kmeans(features, args.num_parts, max_iter=args.max_kmeans_iter)
    labels = np.full(xyz.shape[0], -1, dtype=np.int64)
    labels[fg_idx] = fg_labels
    torch.save(torch.from_numpy(labels), os.path.join(out_dir, "part_labels.pt"))
    sizes = {str(k): int((labels == k).sum()) for k in range(args.num_parts)}
    report = {
        "model_path": os.path.abspath(args.model_path),
        "ply_path": os.path.abspath(ply_path),
        "foreground_mask_path": os.path.abspath(args.foreground_mask_path),
        "num_parts": args.num_parts,
        "foreground_gaussians": int(fg_idx.shape[0]),
        "background_gaussians": int((labels < 0).sum()),
        "cluster_sizes": sizes,
    }
    with open(os.path.join(out_dir, "part_cluster_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_cluster_ply(os.path.join(out_dir, "part_clusters_color.ply"), xyz, labels, args.num_parts)
    print("foreground Gaussians:", report["foreground_gaussians"])
    print("background Gaussians:", report["background_gaussians"])
    print("cluster sizes:")
    for key, val in sizes.items():
        print("  {}: {}".format(key, val))
    print("saved:", os.path.join(out_dir, "part_labels.pt"))
    print("visualization:", os.path.join(out_dir, "part_clusters_color.ply"))


if __name__ == "__main__":
    main()
