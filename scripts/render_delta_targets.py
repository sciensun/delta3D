#!/usr/bin/env python3
"""Render synthetic target views from the same Gaussian bank and cameras."""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def parse_args():
    p = argparse.ArgumentParser()
    lp = ModelParams(p); pp = PipelineParams(p)
    p.add_argument("--delta_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--max_views", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    return p, lp, pp


def save_rgb(path, image):
    array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(array, mode="RGB").save(path)


def main():
    parser, lp, pp = parse_args()
    args = parser.parse_args()
    safe_state(args.quiet)
    payload = torch.load(args.delta_path, map_location="cuda")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(lp.extract(args), gaussians, load_iteration=args.load_iteration, shuffle=False)
    d_xyz = payload["d_xyz"].float().cuda()
    d_scaling = torch.zeros_like(d_xyz)
    d_rotation = torch.zeros((d_xyz.shape[0], 4), device="cuda")
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    cameras = scene.getTrainCameras()[:args.max_views]
    meta = []
    for index, cam in enumerate(cameras):
        package = render(cam, gaussians, pp.extract(args), background, d_xyz, d_rotation, d_scaling, args.is_6dof)
        name = os.path.splitext(os.path.basename(cam.image_name))[0] + ".png"
        save_rgb(os.path.join(args.output_dir, name), package["render"])
        meta.append({"index": index, "image_name": cam.image_name, "output": name})
    with open(os.path.join(args.output_dir, "render_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("rendered views:", len(meta), "to", args.output_dir)


if __name__ == "__main__":
    main()
