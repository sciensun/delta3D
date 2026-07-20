#!/usr/bin/env python3
"""Render source | mined-delta | weak target triptychs for matched views."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--target_image_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=16)
    parser.add_argument("--amplify", type=float, default=1.0)
    parser.add_argument("--composite_target_white", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def resize_like(image, ref):
    if image.shape[-2:] == ref.shape[-2:]:
        return image
    return F.interpolate(image[None], size=ref.shape[-2:], mode="bilinear", align_corners=False)[0]


def to_pil(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def side_by_side(images):
    pil_images = [to_pil(image[:3]) for image in images]
    w, h = pil_images[0].size
    canvas = Image.new("RGB", (w * len(pil_images), h), (255, 255, 255))
    for idx, image in enumerate(pil_images):
        canvas.paste(image, (idx * w, 0))
    return canvas


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    payload = torch.load(args.mined_delta_path, map_location="cuda")
    d_xyz = payload["d_xyz"].to("cuda") * args.amplify
    d_scaling = payload["d_scaling"].to("cuda") * args.amplify
    d_rotation = payload.get("d_rotation", torch.zeros((d_xyz.shape[0], 4))).to("cuda") * args.amplify

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    saved = 0
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            target_path = find_style_target_path(cam, args.target_image_root, required=False)
            if target_path is None:
                continue
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"].clamp(0, 1)
            delta = render(cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)["render"].clamp(0, 1)
            target = load_style_target_rgba(
                cam,
                args.target_image_root,
                device="cuda",
                required=True,
                composite_white=args.composite_target_white,
            )[:3].clamp(0, 1)
            target = resize_like(target, source)
            saved += 1
            side_by_side([source, delta, target]).save(
                os.path.join(args.out_dir, "{:02d}_{}_source_delta_target.png".format(saved, cam.image_name))
            )
            print("{} -> {}".format(cam.image_name, target_path))
            if saved >= args.max_views:
                break
    print("Saved triptychs:", saved)
    print("Output:", args.out_dir)
    print("Panel order: source | delta x{} | target".format(args.amplify))


if __name__ == "__main__":
    main()
