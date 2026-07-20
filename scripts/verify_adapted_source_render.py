#!/usr/bin/env python3
"""Verify an adapted external source PLY through the delta3D renderer."""

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


def load_image_tensor(path, background, device="cuda"):
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    image = image.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
        arr = arr[..., :3] * alpha + bg * (1.0 - alpha)
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def resize_like(image, ref):
    if image.shape[-2:] == ref.shape[-2:]:
        return image
    return F.interpolate(image[None], size=ref.shape[-2:], mode="bilinear", align_corners=False)[0]


def to_pil(tensor):
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def heatmap(abs_diff):
    gray = abs_diff.mean(dim=0, keepdim=True).clamp(0, 1)
    return torch.cat([gray, torch.zeros_like(gray), 1.0 - gray], dim=0)


def side_by_side(images):
    pil_images = [to_pil(image[:3]) for image in images]
    w, h = pil_images[0].size
    canvas = Image.new("RGB", (w * len(pil_images), h), (255, 255, 255))
    for idx, image in enumerate(pil_images):
        canvas.paste(image, (idx * w, 0))
    return canvas


def quantiles(name, values):
    qs = torch.quantile(values.detach().float().cpu(), torch.tensor([0.5, 0.9, 0.95]))
    print(
        "{} mean/median/p90/p95/max: {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}".format(
            name, values.mean().item(), qs[0].item(), qs[1].item(), qs[2].item(), values.max().item()
        )
    )
    return qs


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Verifying adapted external source through delta3D renderer.")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    xyz = gaussians.get_xyz.detach()
    bbox_diag = (xyz.max(dim=0).values - xyz.min(dim=0).values).norm().clamp_min(1e-8)
    scale_norm = gaussians.get_scaling.detach().norm(dim=-1)

    print("loaded iteration:", scene.loaded_iter)
    print("number of Gaussians:", xyz.shape[0])
    print("bbox diagonal: {:.8f}".format(bbox_diag.item()))
    scale_q = quantiles("scale norm", scale_norm)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    per_view = []
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            original_path = None
            for path in candidate_original_paths(args.original_render_root, cam.image_name):
                if os.path.isfile(path):
                    original_path = path
                    break
            if original_path is None:
                continue
            original = load_image_tensor(original_path, bg_color)
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"].clamp(0, 1)
            original = resize_like(original, source)
            diff = (original[:3] - source).abs()
            l1 = diff.mean().item()
            iou = mask_iou(image_to_mask(original), image_to_mask(source)).item()
            per_view.append((cam.image_name, l1, iou))
            side_by_side([original[:3], source, heatmap(diff)]).save(
                os.path.join(args.out_dir, "{:02d}_{}_adapted_source_diff.png".format(len(per_view), cam.image_name))
            )
            print("{} L1={:.6f} maskIoU={:.6f}".format(cam.image_name, l1, iou))
            if len(per_view) >= args.max_views:
                break

    avg_l1 = float(np.mean([x[1] for x in per_view])) if per_view else float("inf")
    avg_iou = float(np.mean([x[2] for x in per_view])) if per_view else 0.0
    scale_p95_ratio = scale_q[2].item() / bbox_diag.item()
    print("average L1: {:.6f}".format(avg_l1))
    print("average foreground mask IoU: {:.6f}".format(avg_iou))
    print("p95 scale / bbox diagonal: {:.6f}".format(scale_p95_ratio))
    print("Interpretation:")
    if not per_view:
        print("FAILED: no matched views found.")
    elif avg_iou >= 0.75 and scale_p95_ratio < 0.03:
        print("LIKELY PASS: inspect saved panels for foreground sharpness. Background differences are acceptable.")
    else:
        print("LIKELY FAIL: if Graphdeco native render is sharp, inspect adapter/loading/camera/background settings.")


if __name__ == "__main__":
    main()
