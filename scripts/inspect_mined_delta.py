#!/usr/bin/env python3
"""Inspect statistics of a mined Stage 1 delta .pt file."""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--zero_eps", type=float, default=1e-5)
    return parser.parse_args()


def stats(name, tensor):
    tensor = tensor.float()
    if tensor.dim() > 1:
        norm = tensor.norm(dim=-1)
    else:
        norm = tensor.abs()
    q = torch.quantile(norm, torch.tensor([0.5, 0.9, 0.95]))
    print("{} norm:".format(name))
    print("  mean   {:.8f}".format(norm.mean().item()))
    print("  median {:.8f}".format(q[0].item()))
    print("  p90    {:.8f}".format(q[1].item()))
    print("  p95    {:.8f}".format(q[2].item()))
    print("  max    {:.8f}".format(norm.max().item()))
    return norm


def main():
    args = parse_args()
    payload = torch.load(args.mined_delta_path, map_location="cpu")
    print("File:", args.mined_delta_path)
    print("Keys:", sorted(payload.keys()))

    source_xyz = payload.get("source_xyz")
    if source_xyz is None:
        raise RuntimeError("Missing source_xyz in mined delta file.")
    source_xyz = source_xyz.float()
    bbox_min = source_xyz.min(dim=0).values
    bbox_max = source_xyz.max(dim=0).values
    bbox_diag = (bbox_max - bbox_min).norm()
    print("Number of Gaussians:", source_xyz.shape[0])
    print("source_xyz bbox min:", bbox_min.tolist())
    print("source_xyz bbox max:", bbox_max.tolist())
    print("bbox diagonal: {:.8f}".format(bbox_diag.item()))

    d_xyz = payload.get("d_xyz")
    d_scaling = payload.get("d_scaling")
    if d_xyz is None:
        raise RuntimeError("Missing d_xyz in mined delta file.")
    if d_scaling is None:
        raise RuntimeError("Missing d_scaling in mined delta file.")

    d_xyz_norm = stats("d_xyz", d_xyz)
    d_scaling_norm = stats("d_scaling", d_scaling)
    max_abs_ratio = d_xyz.float().abs().max() / bbox_diag.clamp_min(1e-8)
    print("max |d_xyz| / bbox diagonal: {:.8f}".format(max_abs_ratio.item()))
    print("d_xyz nearly zero:", bool(d_xyz_norm.max().item() < args.zero_eps))
    print("d_scaling nearly zero:", bool(d_scaling_norm.max().item() < args.zero_eps))


if __name__ == "__main__":
    main()
