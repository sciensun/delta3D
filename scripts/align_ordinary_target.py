#!/usr/bin/env python3
"""Estimate rigid + preferred-uniform-scale alignment for two point sets.

This is a coordinate cleanup step for a unified ordinary Gaussian/point
representation. It does not non-rigidly warp the target and does not replace
the 3DGS representation.
"""
import argparse
import json
import os

import torch


def load_points(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("xyz", "source_xyz", "target_xyz"):
            if key in payload:
                return payload[key].float()
        raise ValueError("{} has no xyz/source_xyz/target_xyz tensor".format(path))
    return payload.float()


def fit_uniform_similarity(source, target):
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have identical [N,3] shapes")
    source_center = source.mean(0); target_center = target.mean(0)
    xs = source - source_center; yt = target - target_center
    covariance = yt.T @ xs / max(1, source.shape[0])
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3)
    if torch.det(u @ vh) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vh
    scale = singular.mul(torch.diag(correction)).sum() / xs.square().sum().clamp_min(1e-8)
    scale = scale.clamp_min(1e-8)
    translation = target_center - scale * (rotation @ source_center)
    aligned = scale * (source @ rotation.T) + translation
    return aligned, rotation, scale, translation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_xyz_path", required=True)
    p.add_argument("--target_xyz_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--transform_path", required=True)
    a = p.parse_args()
    source, target = load_points(a.source_xyz_path), load_points(a.target_xyz_path)
    aligned, rotation, scale, translation = fit_uniform_similarity(source, target)
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    torch.save({"xyz": aligned, "source_xyz": source, "target_xyz": target,
                "metadata": {"method": "uniform_similarity_umeyama", "nonrigid_warp": False}}, a.output_path)
    transform = {"method": "uniform_similarity_umeyama", "scale": float(scale),
                 "rotation": rotation.tolist(), "translation": translation.tolist(),
                 "before_rmse": float((source - target).square().mean().sqrt()),
                 "after_rmse": float((aligned - target).square().mean().sqrt())}
    with open(a.transform_path, "w", encoding="utf-8") as f: json.dump(transform, f, indent=2)
    print(json.dumps(transform, indent=2))


if __name__ == "__main__": main()
