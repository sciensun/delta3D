#!/usr/bin/env python3
"""Render mined delta with amplification factors for visual debugging."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--amplify", nargs="+", type=float, default=[1, 2, 5, 10])
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def to_pil(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype("uint8"))


def side_by_side(images):
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for idx, image in enumerate(images):
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
    d_xyz = payload["d_xyz"].to("cuda")
    d_scaling = payload["d_scaling"].to("cuda")
    d_rotation = payload.get("d_rotation", torch.zeros((d_xyz.shape[0], 4))).to("cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for idx, cam in enumerate(scene.getTrainCameras()[: args.max_views]):
            panels = [to_pil(render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"])]
            for factor in args.amplify:
                image = render(
                    cam,
                    gaussians,
                    pipe,
                    background,
                    d_xyz * factor,
                    d_rotation * factor,
                    d_scaling * factor,
                    dataset.is_6dof,
                )["render"]
                panels.append(to_pil(image))
            side_by_side(panels).save(
                os.path.join(args.out_dir, "{:02d}_{}_source_amp.png".format(idx + 1, cam.image_name))
            )

    print("Saved amplified mined delta comparisons to:", args.out_dir)
    print("Panel order: source | " + " | ".join("x{}".format(v) for v in args.amplify))


if __name__ == "__main__":
    main()
