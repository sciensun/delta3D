#!/usr/bin/env python3
"""Prune source Gaussians whose centers project outside GLB foreground masks.

This is a source 3DGS cleanup utility before Stage 1. It does not use ChatGPT
targets and does not modify the research delta pipeline.
"""

import argparse
import json
import math
import os
import shutil
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData, PlyElement


EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model_path", required=True)
    parser.add_argument("--source_dataset_path", required=True)
    parser.add_argument("--out_model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--include_test", action="store_true", default=True)
    parser.add_argument("--max_views", type=int, default=216)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--min_projected_views", type=int, default=8)
    parser.add_argument("--min_fg_ratio", type=float, default=0.2)
    parser.add_argument("--min_fg_hits", type=int, default=2)
    parser.add_argument("--copy_metadata", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_frames(dataset_path, include_test=True):
    for split, name in (("train", "transforms_train.json"), ("test", "transforms_test.json")):
        if split == "test" and not include_test:
            continue
        path = os.path.join(dataset_path, name)
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        common = {
            "camera_angle_x": payload.get("camera_angle_x"),
            "camera_angle_y": payload.get("camera_angle_y", payload.get("camera_angle_x")),
            "w": payload.get("w"),
            "h": payload.get("h"),
        }
        for frame in payload.get("frames", []):
            item = dict(frame)
            item.update(common)
            item["split"] = split
            yield item


def find_image(dataset_path, file_path):
    root, ext = os.path.splitext(os.path.join(dataset_path, file_path))
    candidates = [root + ext] if ext else []
    candidates.extend(root + suffix for suffix in EXTS)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not find image for frame {}".format(file_path))


def foreground_mask(path, threshold=0.5, dilate=9):
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_alpha:
        alpha = image.convert("RGBA").split()[-1]
        mask = alpha.point(lambda v: 255 if v / 255.0 > threshold else 0)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        dist_white = np.abs(rgb - 1.0).mean(axis=-1)
        mask = Image.fromarray((dist_white > 0.08).astype(np.uint8) * 255, "L")
    if dilate > 1:
        if dilate % 2 == 0:
            dilate += 1
        mask = mask.filter(ImageFilter.MaxFilter(dilate))
    return np.asarray(mask, dtype=np.uint8) > 0


def project_points(xyz, c2w, fovx, fovy, width, height):
    w2c = np.linalg.inv(c2w)
    points_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
    cam = points_h @ w2c.T
    z = cam[:, 2]
    front = z < -1e-6
    denom = np.maximum(-z, 1e-6)
    x_ndc = (cam[:, 0] / denom) / math.tan(float(fovx) * 0.5)
    y_ndc = (cam[:, 1] / denom) / math.tan(float(fovy) * 0.5)
    u = (x_ndc + 1.0) * 0.5 * width
    v = (1.0 - y_ndc) * 0.5 * height
    in_bounds = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u.astype(np.int64), v.astype(np.int64), in_bounds


def main():
    args = parse_args()
    src_ply = os.path.join(
        args.source_model_path,
        "point_cloud",
        "iteration_{}".format(args.iteration),
        "point_cloud.ply",
    )
    dst_ply = os.path.join(
        args.out_model_path,
        "point_cloud",
        "iteration_{}".format(args.iteration),
        "point_cloud.ply",
    )
    if not os.path.isfile(src_ply):
        raise FileNotFoundError(src_ply)
    if os.path.exists(dst_ply) and not args.force:
        raise FileExistsError("{} exists; use --force to overwrite.".format(dst_ply))

    ply = PlyData.read(src_ply)
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    projected = np.zeros(xyz.shape[0], dtype=np.int32)
    fg_hits = np.zeros(xyz.shape[0], dtype=np.int32)

    frames = list(iter_frames(args.source_dataset_path, include_test=args.include_test))[: args.max_views]
    if not frames:
        raise RuntimeError("No frames found in {}".format(args.source_dataset_path))

    for idx, frame in enumerate(frames, 1):
        image_path = find_image(args.source_dataset_path, frame["file_path"])
        mask = foreground_mask(image_path, threshold=args.mask_threshold, dilate=args.mask_dilate)
        height, width = mask.shape
        fovx = frame["camera_angle_x"]
        fovy = frame.get("camera_angle_y") or frame["camera_angle_x"]
        u, v, in_bounds = project_points(
            xyz,
            np.asarray(frame["transform_matrix"], dtype=np.float64),
            fovx,
            fovy,
            width,
            height,
        )
        projected += in_bounds.astype(np.int32)
        valid_idx = np.where(in_bounds)[0]
        if valid_idx.size:
            fg = mask[v[valid_idx], u[valid_idx]]
            fg_hits[valid_idx] += fg.astype(np.int32)
        if idx % 25 == 0 or idx == len(frames):
            print("processed masks: {}/{}".format(idx, len(frames)))

    fg_ratio = fg_hits / np.maximum(projected, 1)
    enough_views = projected >= args.min_projected_views
    keep = (~enough_views) | (fg_hits >= args.min_fg_hits) | (fg_ratio >= args.min_fg_ratio)
    pruned = ~keep

    os.makedirs(os.path.dirname(dst_ply), exist_ok=True)
    if os.path.exists(dst_ply):
        os.unlink(dst_ply)
    out_vertex = vertex[keep]
    PlyData([PlyElement.describe(out_vertex, "vertex")], text=ply.text).write(dst_ply)

    if args.copy_metadata:
        for name in ("source_builder_graphdeco.json", "README_adapted_source.md", "cfg_args"):
            src = os.path.join(args.source_model_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out_model_path, name))

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_model_path": os.path.abspath(args.source_model_path),
        "source_dataset_path": os.path.abspath(args.source_dataset_path),
        "out_model_path": os.path.abspath(args.out_model_path),
        "iteration": args.iteration,
        "input_gaussians": int(xyz.shape[0]),
        "kept_gaussians": int(keep.sum()),
        "pruned_gaussians": int(pruned.sum()),
        "kept_ratio": float(keep.mean()),
        "frames_used": len(frames),
        "mask_threshold": args.mask_threshold,
        "mask_dilate": args.mask_dilate,
        "min_projected_views": args.min_projected_views,
        "min_fg_ratio": args.min_fg_ratio,
        "min_fg_hits": args.min_fg_hits,
        "projected_views_mean": float(projected.mean()),
        "fg_ratio_mean": float(fg_ratio.mean()),
        "fg_ratio_p05": float(np.quantile(fg_ratio, 0.05)),
        "fg_ratio_p50": float(np.quantile(fg_ratio, 0.50)),
        "fg_ratio_p95": float(np.quantile(fg_ratio, 0.95)),
    }
    os.makedirs(args.out_model_path, exist_ok=True)
    with open(os.path.join(args.out_model_path, "source_foreground_prune_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Source foreground pruning complete.")
    print("Input Gaussians:", report["input_gaussians"])
    print("Kept Gaussians:", report["kept_gaussians"])
    print("Pruned Gaussians:", report["pruned_gaussians"])
    print("Kept ratio: {:.4f}".format(report["kept_ratio"]))
    print("Output PLY:", dst_ply)
    print("Report:", os.path.join(args.out_model_path, "source_foreground_prune_report.json"))


if __name__ == "__main__":
    main()
