#!/usr/bin/env python3
"""Render source vs several forced deformations to verify renderer delta path."""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--force_dx", type=float, default=0.2)
    parser.add_argument("--force_dy", type=float, default=0.0)
    parser.add_argument("--force_dz", type=float, default=0.0)
    parser.add_argument("--radial_strength", type=float, default=0.15)
    parser.add_argument("--force_scale", type=float, default=0.1)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def to_pil(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def save_panels(panels, out_dir, stem):
    names = [
        "01_source",
        "02_translation",
        "03_radial_expand",
        "04_radial_shrink",
        "05_scale_expand",
        "06_scale_shrink",
    ]
    pil_images = []
    for name, tensor in zip(names, panels):
        image = to_pil(tensor)
        image.save(os.path.join(out_dir, "{}_{}.png".format(stem, name)))
        pil_images.append(image)

    w, h = pil_images[0].size
    canvas = Image.new("RGB", (w * len(pil_images), h), (255, 255, 255))
    for idx, image in enumerate(pil_images):
        canvas.paste(image, (idx * w, 0))
    canvas.save(os.path.join(out_dir, "{}_side_by_side.png".format(stem)))


def norm_stats(name, tensor):
    norms = tensor.norm(dim=-1)
    print("{} mean norm: {:.8f}".format(name, norms.mean().item()))
    print("{} max norm:  {:.8f}".format(name, norms.max().item()))


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    xyz = gaussians.get_xyz
    n = xyz.shape[0]
    center = xyz.detach().mean(dim=0, keepdim=True)

    translation = torch.tensor([args.force_dx, args.force_dy, args.force_dz], device="cuda").view(1, 3).expand(n, 3)
    radial_expand = args.radial_strength * (xyz.detach() - center)
    radial_shrink = -radial_expand
    scale_expand = torch.full((n, 3), args.force_scale, device="cuda")
    scale_shrink = -scale_expand
    zero_xyz = torch.zeros_like(translation)
    zero_scaling = torch.zeros_like(scale_expand)
    d_rotation = torch.zeros((n, 4), device="cuda")

    print("Forced delta magnitudes:")
    norm_stats("translation d_xyz", translation)
    norm_stats("radial expand d_xyz", radial_expand)
    norm_stats("radial shrink d_xyz", radial_shrink)
    norm_stats("scale expand d_scaling", scale_expand)
    norm_stats("scale shrink d_scaling", scale_shrink)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    diff_radial = []
    diff_scale = []

    with torch.no_grad():
        for idx, cam in enumerate(scene.getTrainCameras()[: args.max_views]):
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"]
            translated = render(cam, gaussians, pipe, background, translation, d_rotation, zero_scaling, dataset.is_6dof)["render"]
            radial_out = render(cam, gaussians, pipe, background, radial_expand, d_rotation, zero_scaling, dataset.is_6dof)["render"]
            radial_in = render(cam, gaussians, pipe, background, radial_shrink, d_rotation, zero_scaling, dataset.is_6dof)["render"]
            scale_out = render(cam, gaussians, pipe, background, zero_xyz, d_rotation, scale_expand, dataset.is_6dof)["render"]
            scale_in = render(cam, gaussians, pipe, background, zero_xyz, d_rotation, scale_shrink, dataset.is_6dof)["render"]

            diff_radial.append((source - radial_out).abs().mean().item())
            diff_scale.append((source - scale_out).abs().mean().item())
            save_panels(
                [source, translated, radial_out, radial_in, scale_out, scale_in],
                args.out_dir,
                "{:02d}_{}".format(idx + 1, cam.image_name),
            )

    mean_radial = float(np.mean(diff_radial)) if diff_radial else 0.0
    mean_scale = float(np.mean(diff_scale)) if diff_scale else 0.0
    print("Saved forced delta comparisons to:", args.out_dir)
    print("Mean pixel L1(source, radial expansion): {:.8f}".format(mean_radial))
    print("Mean pixel L1(source, scaling expansion): {:.8f}".format(mean_scale))
    if mean_radial < 1e-4 and mean_scale < 1e-4:
        print("WARNING: Renderer may not be applying deformation or image saving may be wrong.")


if __name__ == "__main__":
    main()
