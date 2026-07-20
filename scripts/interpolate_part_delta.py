#!/usr/bin/env python3
"""Render alpha interpolation for a fitted part delta."""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.style_image_utils import find_style_target_path, load_style_target_rgba


def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--part_delta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_image_root", default=None)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def to_pil(tensor):
    image = tensor.detach().clamp(0, 1)[:3].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((image * 255.0).astype("uint8"))


def make_panel(images):
    widths, heights = zip(*(image.size for image in images))
    canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.size[0]
    return canvas


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)
    payload = torch.load(args.part_delta_path, map_location="cuda")
    d_xyz = payload["d_xyz"].to("cuda")
    d_scaling = payload.get("d_scaling", torch.zeros_like(d_xyz)).to("cuda")
    d_rotation = payload.get("d_rotation", torch.zeros((d_xyz.shape[0], 4))).to("cuda")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    saved = 0
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            if args.target_image_root and find_style_target_path(cam, args.target_image_root, required=False) is None:
                continue
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"].clamp(0, 1)
            views = [to_pil(source)]
            alpha_images = []
            for alpha in args.alphas:
                alpha_dir = os.path.join(args.out_dir, "alpha_{:03d}".format(int(round(alpha * 100))))
                os.makedirs(alpha_dir, exist_ok=True)
                image = render(
                    cam,
                    gaussians,
                    pipe,
                    background,
                    d_xyz * alpha,
                    d_rotation * alpha,
                    d_scaling * alpha,
                    dataset.is_6dof,
                )["render"].clamp(0, 1)
                image_pil = to_pil(image)
                image_pil.save(os.path.join(alpha_dir, "{}_alpha_{:03d}.png".format(cam.image_name, int(round(alpha * 100)))))
                alpha_images.append(image_pil)
            panel_images = [views[0]] + alpha_images
            if args.target_image_root:
                target = load_style_target_rgba(cam, args.target_image_root, device="cuda", required=True, composite_white=True)
                panel_images.append(to_pil(target))
            make_panel(panel_images).save(os.path.join(args.out_dir, "{:02d}_{}_interpolation.png".format(saved + 1, cam.image_name)))
            saved += 1
            if saved >= args.max_views:
                break
    print("saved views:", saved)
    print("output:", args.out_dir)
    print("panel order: source | " + " | ".join("alpha {:.2f}".format(a) for a in args.alphas) + (" | target" if args.target_image_root else ""))


if __name__ == "__main__":
    main()
