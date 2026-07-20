#!/usr/bin/env python3
"""Estimate object/foreground Gaussian mask from source cameras and silhouettes."""

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from plyfile import PlyData


EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_dilate", type=int, default=7)
    parser.add_argument("--max_views", type=int, default=216)
    parser.add_argument("--include_test", action="store_true", default=True)
    return parser.parse_args()


def load_frames(source_path, include_test=True):
    frames = []
    for split, name in (("train", "transforms_train.json"), ("test", "transforms_test.json")):
        if split == "test" and not include_test:
            continue
        path = os.path.join(source_path, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for frame in payload.get("frames", []):
            item = dict(frame)
            item["split"] = split
            item["camera_angle_x"] = payload["camera_angle_x"]
            item["camera_angle_y"] = payload.get("camera_angle_y", payload["camera_angle_x"])
            frames.append(item)
    return frames


def find_image(source_path, file_path):
    root, ext = os.path.splitext(os.path.join(source_path, file_path))
    candidates = [root + ext] if ext else []
    candidates.extend(root + suffix for suffix in EXTS)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(file_path)


def image_mask(path, dilate=7):
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_alpha:
        mask = image.convert("RGBA").split()[-1].point(lambda v: 255 if v > 25 else 0)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mask = Image.fromarray((np.abs(rgb - 1.0).mean(axis=-1) > 0.08).astype(np.uint8) * 255, "L")
    if dilate > 1:
        if dilate % 2 == 0:
            dilate += 1
        mask = mask.filter(ImageFilter.MaxFilter(dilate))
    return np.asarray(mask, dtype=np.uint8) > 0


def load_xyz(model_path, iteration):
    ply_path = os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    return xyz, ply_path


def project(xyz, c2w, fovx, fovy, width, height):
    w2c = np.linalg.inv(c2w)
    pts = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1) @ w2c.T
    z = pts[:, 2]
    front = z < -1e-6
    denom = np.maximum(-z, 1e-6)
    x_ndc = (pts[:, 0] / denom) / math.tan(float(fovx) * 0.5)
    y_ndc = (pts[:, 1] / denom) / math.tan(float(fovy) * 0.5)
    u = ((x_ndc + 1.0) * 0.5 * width).astype(np.int64)
    v = ((1.0 - y_ndc) * 0.5 * height).astype(np.int64)
    valid = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def main():
    args = parse_args()
    out_dir = args.out_dir or args.model_path
    os.makedirs(out_dir, exist_ok=True)
    xyz, ply_path = load_xyz(args.model_path, args.load_iteration)
    total_visible = np.zeros(xyz.shape[0], dtype=np.int32)
    visible_inside = np.zeros(xyz.shape[0], dtype=np.int32)
    frames = load_frames(args.source_path, include_test=args.include_test)[: args.max_views]
    for idx, frame in enumerate(frames, 1):
        mask = image_mask(find_image(args.source_path, frame["file_path"]), dilate=args.mask_dilate)
        h, w = mask.shape
        u, v, valid = project(
            xyz,
            np.asarray(frame["transform_matrix"], dtype=np.float64),
            frame["camera_angle_x"],
            frame["camera_angle_y"],
            w,
            h,
        )
        total_visible += valid.astype(np.int32)
        valid_idx = np.where(valid)[0]
        if valid_idx.size:
            visible_inside[valid_idx] += mask[v[valid_idx], u[valid_idx]].astype(np.int32)
        if idx % 25 == 0 or idx == len(frames):
            print("processed views: {}/{}".format(idx, len(frames)))

    support = visible_inside / np.maximum(total_visible, 1)
    foreground_mask = support >= args.threshold
    torch.save(torch.from_numpy(support.astype(np.float32)), os.path.join(out_dir, "foreground_support.pt"))
    torch.save(torch.from_numpy(foreground_mask.astype(np.bool_)), os.path.join(out_dir, "foreground_mask.pt"))
    hist, edges = np.histogram(support, bins=np.linspace(0.0, 1.0, 11))
    report = {
        "ply_path": os.path.abspath(ply_path),
        "source_path": os.path.abspath(args.source_path),
        "total_gaussians": int(xyz.shape[0]),
        "foreground_gaussians": int(foreground_mask.sum()),
        "background_gaussians": int((~foreground_mask).sum()),
        "threshold": args.threshold,
        "mask_dilate": args.mask_dilate,
        "views_used": len(frames),
        "histogram_bins": edges.tolist(),
        "histogram_counts": hist.tolist(),
    }
    with open(os.path.join(out_dir, "foreground_mask_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("total Gaussians:", report["total_gaussians"])
    print("foreground Gaussians:", report["foreground_gaussians"])
    print("background Gaussians:", report["background_gaussians"])
    print("histogram counts:", report["histogram_counts"])
    print("saved:", os.path.join(out_dir, "foreground_mask.pt"))


if __name__ == "__main__":
    main()
