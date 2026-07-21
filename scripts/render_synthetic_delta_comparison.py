#!/usr/bin/env python3
"""Render source | known teacher | recovered teacher for synthetic benchmark."""
import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def save_panel(path, images):
    width, height = images[0].size
    panel = Image.new("RGB", (width * len(images), height), "white")
    for i, image in enumerate(images): panel.paste(image, (i * width, 0))
    panel.save(path)


def tensor_image(x):
    return Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


def main():
    p = argparse.ArgumentParser()
    lp = ModelParams(p); pp = PipelineParams(p)
    p.add_argument("--ground_truth_path", required=True); p.add_argument("--recovered_path", required=True)
    p.add_argument("--output_dir", required=True); p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--max_views", type=int, default=8)
    args = p.parse_args(); safe_state(False)
    gt = torch.load(args.ground_truth_path, map_location="cuda"); rec = torch.load(args.recovered_path, map_location="cuda")
    gaussians = GaussianModel(args.sh_degree); scene = Scene(lp.extract(args), gaussians, load_iteration=args.load_iteration, shuffle=False)
    pipe = pp.extract(args); bg = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], device="cuda", dtype=torch.float32)
    os.makedirs(args.output_dir, exist_ok=True)
    zero = torch.zeros_like(gt["d_xyz"], device="cuda"); zero_rot = torch.zeros((zero.shape[0], 4), device="cuda")
    for cam in scene.getTrainCameras()[:args.max_views]:
        def render_delta(payload):
            return tensor_image(render(cam, gaussians, pipe, bg, payload["d_xyz"].cuda(), zero_rot, zero, args.is_6dof)["render"])
        source = render_delta({"d_xyz": zero}); known = render_delta(gt); recovered = render_delta(rec)
        name = os.path.splitext(os.path.basename(cam.image_name))[0] + "_source_known_recovered.png"
        save_panel(os.path.join(args.output_dir, name), [source, known, recovered])
    print("saved synthetic comparison panels:", args.output_dir)


if __name__ == "__main__": main()
