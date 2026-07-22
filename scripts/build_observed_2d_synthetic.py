#!/usr/bin/env python3
"""Build image-first synthetic ObservationBundle files.

Ground-truth xyz is used only here to generate projected observations. It is
never written to the observed_2d bundle consumed by Stage 1.
"""
import argparse
import json
import os
import random
import shutil
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams
from correspondence.schema import ObservationBundle
from gaussian_renderer import render  # noqa: F401, keeps the repo CUDA extension import path explicit
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def parse_indices(value):
    return [int(x) for x in value.split(",") if x.strip()]


def project(points, camera):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    clip = torch.cat([points, ones], dim=1) @ camera.full_proj_transform
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None].clamp_min(1e-8)
    xy = torch.stack([
        (ndc[:, 0] * 0.5 + 0.5) * float(camera.image_width),
        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(camera.image_height),
    ], dim=1)
    visible = (w > 0) & (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1)
    return xy, visible


def target_image_for_name(root, image_name):
    stem = os.path.basename(str(image_name))
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(root, stem + extension)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("no synthetic target image for {} under {}".format(image_name, root))


def build_bundle(source_path, model_path, gt_delta_path, foreground_mask_path,
                 target_image_root, output_bundle, output_target_root,
                 view_indices, noise_std=0.0, outlier_rate=0.0,
                 visibility_keep=1.0, seed=0, load_iteration=30000):
    random.seed(seed)
    torch.manual_seed(seed)
    payload = torch.load(gt_delta_path, map_location="cpu")
    # This tensor is used only for projection and is deliberately not included
    # in the serialized observed_2d bundle.
    target_xyz = payload["source_xyz"].float() + payload["d_xyz"].float()
    source_xyz = payload["source_xyz"].float()
    foreground = torch.load(foreground_mask_path, map_location="cpu").bool().flatten()
    if len(source_xyz) != len(foreground):
        raise ValueError("foreground mask does not match source Gaussian count")

    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    args = parser.parse_args(["-s", source_path, "--model_path", model_path])
    safe_state(True)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(lp.extract(args), gaussians, load_iteration=load_iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    if max(view_indices) >= len(cameras):
        raise IndexError("view index exceeds source camera count")

    target_xy = []
    visibility = []
    names = []
    for view_index in view_indices:
        camera = cameras[view_index]
        source_xy, source_in_frame = project(source_xyz.cuda(), camera)
        target_view_xy, target_in_frame = project(target_xyz.cuda(), camera)
        visible = source_in_frame & target_in_frame & foreground.cuda()
        if visibility_keep < 1.0:
            keep = torch.rand(len(visible), device="cuda") < visibility_keep
            visible = visible & keep
        observed = target_view_xy.clone()
        if noise_std > 0:
            observed[visible] += torch.randn_like(observed[visible]) * float(noise_std)
        if outlier_rate > 0 and visible.any():
            outlier = visible & (torch.rand(len(visible), device="cuda") < outlier_rate)
            observed[outlier, 0] = torch.rand(int(outlier.sum()), device="cuda") * float(camera.image_width)
            observed[outlier, 1] = torch.rand(int(outlier.sum()), device="cuda") * float(camera.image_height)
        target_xy.append(observed.detach().cpu())
        visibility.append(visible.detach().cpu())
        names.append(camera.image_name)
        os.makedirs(output_target_root, exist_ok=True)
        shutil.copy2(target_image_for_name(target_image_root, camera.image_name),
                     os.path.join(output_target_root, os.path.basename(target_image_for_name(target_image_root, camera.image_name))))

    target_xy = torch.stack(target_xy)
    visibility = torch.stack(visibility)
    support = visibility.sum(0).long()
    confidence = visibility.float()
    bundle = ObservationBundle(
        source_xyz=source_xyz,
        target_xy=target_xy,
        visibility_2d=visibility,
        confidence_2d=confidence,
        support_count_2d=support,
        camera_names=names,
        observation_mode="observed_2d",
        metadata={
            "benchmark": "synthetic_observed_2d",
            "teacher": "body_roundness",
            "seed": int(seed),
            "noise_std_pixels": float(noise_std),
            "outlier_rate": float(outlier_rate),
            "visibility_keep": float(visibility_keep),
            "view_indices": [int(x) for x in view_indices],
            "target_xyz_in_optimizer_input": False,
        },
    )
    bundle.save(output_bundle)
    with open(os.path.splitext(output_bundle)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(bundle.metadata, handle, indent=2)
    print("saved observed_2d bundle:", output_bundle)
    print("views:", names, "support:", int((support > 0).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--gt_delta_path", required=True)
    p.add_argument("--foreground_mask_path", required=True)
    p.add_argument("--target_image_root", required=True)
    p.add_argument("--output_bundle", required=True)
    p.add_argument("--output_target_root", required=True)
    p.add_argument("--view_indices", required=True)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--outlier_rate", type=float, default=0.0)
    p.add_argument("--visibility_keep", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load_iteration", type=int, default=30000)
    a = p.parse_args()
    build_bundle(a.source_path, a.model_path, a.gt_delta_path, a.foreground_mask_path,
                 a.target_image_root, a.output_bundle, a.output_target_root,
                 parse_indices(a.view_indices), a.noise_std, a.outlier_rate,
                 a.visibility_keep, a.seed, a.load_iteration)


if __name__ == "__main__":
    main()
