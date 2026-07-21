#!/usr/bin/env python3
"""Deprecated compatibility wrapper for already paired point sets.

Use ``fit_similarity_from_corresponded_points.py`` for paired points or
``align_ordinary_reference.py`` for an independent target with sparse anchors.
"""
import argparse
import json
import os
import sys

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.alignment import fit_similarity_from_corresponded_points, apply_similarity, transform_to_json


def load_points(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("xyz", "source_xyz", "target_xyz"):
            if key in payload:
                return payload[key].float()
        raise ValueError("{} has no xyz/source_xyz/target_xyz tensor".format(path))
    return payload.float()


def fit_uniform_similarity(source, target):
    transform = fit_similarity_from_corresponded_points(source, target)
    return apply_similarity(source, transform), transform["rotation"], transform["scale"], transform["translation"]


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
