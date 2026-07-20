#!/usr/bin/env python3
"""Render source-only diagnostics against GLB foreground masks."""

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
from utils.mask_utils import image_to_mask, mask_iou


EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--original_render_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def candidate_original_paths(root, image_name):
    stem, _ = os.path.splitext(image_name)
    base = os.path.basename(stem)
    names = [image_name]
    names.extend(stem + ext for ext in EXTS)
    names.extend(base + ext for ext in EXTS)
    return [os.path.join(root, name) for name in dict.fromkeys(names)]


def load_rgba(path, bg_color, device="cuda"):
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    alpha = arr[..., 3:4]
    bg = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3)
    rgb = arr[..., :3] * alpha + bg * (1.0 - alpha)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return torch.from_numpy(rgba).permute(2, 0, 1).to(device)


def to_pil_rgb(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray((arr[..., :3] * 255.0).astype(np.uint8))


def side_by_side(images):
    pil_images = [to_pil_rgb(image) for image in images]
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
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rows = []
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            original_path = None
            for path in candidate_original_paths(args.original_render_root, cam.image_name):
                if os.path.isfile(path):
                    original_path = path
                    break
            if original_path is None:
                continue
            original = load_rgba(original_path, bg_color)
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"].clamp(0, 1)
            orig_mask = image_to_mask(original)
            src_mask = image_to_mask(source)
            outside = (src_mask * (1.0 - orig_mask)).clamp(0, 1)
            iou = mask_iou(orig_mask, src_mask).item()
            outside_ratio = outside.sum().item() / src_mask.sum().clamp_min(1.0).item()
            rows.append((cam.image_name, iou, outside_ratio))
            side_by_side([original[:3], source, orig_mask.repeat(3, 1, 1), src_mask.repeat(3, 1, 1), outside.repeat(3, 1, 1)]).save(
                os.path.join(args.out_dir, "{:02d}_{}_source_foreground_diag.png".format(len(rows), cam.image_name))
            )
            print("{} maskIoU={:.6f} outsideSourceMaskRatio={:.6f}".format(cam.image_name, iou, outside_ratio))
            if len(rows) >= args.max_views:
                break
    if rows:
        print("average mask IoU: {:.6f}".format(float(np.mean([x[1] for x in rows]))))
        print("average outside source-mask ratio: {:.6f}".format(float(np.mean([x[2] for x in rows]))))
    print("Panel order: GLB rgb | source rgb | GLB mask | source mask | source outside GLB mask")


if __name__ == "__main__":
    main()
