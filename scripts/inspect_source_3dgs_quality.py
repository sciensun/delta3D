#!/usr/bin/env python3
"""Inspect basic quality indicators of a trained source 3DGS point cloud."""

import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.system_utils import searchForMaxIteration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    return parser.parse_args()


def resolve_ply(model_path, iteration):
    pc_root = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(pc_root)
    path = os.path.join(pc_root, "iteration_{}".format(iteration), "point_cloud.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return iteration, path


def quantiles(name, values):
    values = torch.as_tensor(values, dtype=torch.float32)
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95]))
    print("{}:".format(name))
    print("  mean   {:.8f}".format(values.mean().item()))
    print("  median {:.8f}".format(q[0].item()))
    print("  p90    {:.8f}".format(q[1].item()))
    print("  p95    {:.8f}".format(q[2].item()))
    print("  max    {:.8f}".format(values.max().item()))
    return values


def main():
    args = parse_args()
    iteration, path = resolve_ply(args.model_path, args.load_iteration)
    ply = PlyData.read(path)
    vertex = ply.elements[0]

    xyz = np.stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1)
    scale_names = sorted([p.name for p in vertex.properties if p.name.startswith("scale_")])
    scales_raw = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
    scales = np.exp(scales_raw)
    opacity_raw = np.asarray(vertex["opacity"])
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    scale_norm = np.linalg.norm(scales, axis=1)

    f_rest = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    active_sh_hint = int(round((len(f_rest) / 3 + 1) ** 0.5 - 1)) if f_rest else 0

    print("model_path:", args.model_path)
    print("loaded iteration:", iteration)
    print("point_cloud:", path)
    print("number of Gaussians:", xyz.shape[0])
    print("xyz bbox min:", bbox_min.tolist())
    print("xyz bbox max:", bbox_max.tolist())
    print("xyz bbox diagonal: {:.8f}".format(bbox_diag))
    quantiles("Gaussian scale norm", scale_norm)
    quantiles("Opacity", opacity)
    print("active/max SH degree hint from PLY:", active_sh_hint)

    if xyz.shape[0] < 20000:
        print("WARNING: Gaussian count is < 20k; source may be underfit for a detailed object.")
    if bbox_diag > 0 and np.percentile(scale_norm, 95) / bbox_diag > 0.03:
        print("WARNING: p95 Gaussian scale is large relative to bbox; render may be blurry.")
    if float(np.mean(opacity)) < 0.05:
        print("WARNING: mean opacity is very low; source may be too diffuse or under-trained.")
    if float(np.percentile(opacity, 95)) < 0.2:
        print("WARNING: p95 opacity is low; opacity distribution may be too diffuse.")


if __name__ == "__main__":
    main()
