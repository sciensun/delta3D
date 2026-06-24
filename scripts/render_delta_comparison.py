#!/usr/bin/env python3
"""Render source vs mined-delta vs weak target comparison images."""

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
from utils.style_image_utils import find_style_target_path, load_style_target_rgba, match_image_size


def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--target_image_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def to_pil(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def make_triptych(source, delta, target):
    h, w = source.shape[-2:]
    target = match_image_size(target, source)
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    canvas.paste(to_pil(source), (0, 0))
    canvas.paste(to_pil(delta), (w, 0))
    canvas.paste(to_pil(target[:3]), (w * 2, 0))
    return canvas


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    out_dir = args.out_dir or os.path.join(args.model_path, "delta_comparison")
    os.makedirs(out_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    payload = torch.load(args.mined_delta_path, map_location="cuda")
    d_xyz = payload["d_xyz"].to("cuda")
    d_scaling = payload["d_scaling"].to("cuda")
    d_rotation = payload.get("d_rotation", torch.zeros((d_xyz.shape[0], 4))).to("cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    saved = 0
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            target_path = find_style_target_path(cam, args.target_image_root, required=False)
            if target_path is None:
                continue
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"]
            delta = render(cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)["render"]
            target = load_style_target_rgba(cam, args.target_image_root, device="cuda", required=True)
            triptych = make_triptych(source, delta, target)
            triptych.save(os.path.join(out_dir, "{:02d}_{}_source_delta_target.png".format(saved + 1, cam.image_name)))
            saved += 1
            if saved >= args.max_views:
                break

    print("Saved {} comparison images to {}".format(saved, out_dir))
    if saved == 0:
        print("No matching target views found. Check target names and source view names.")


if __name__ == "__main__":
    main()
