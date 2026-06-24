#!/usr/bin/env python3
"""Source 3DGS quality gate: compare perspective GLB renders against source 3DGS."""

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
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def quantiles(name, values):
    q = torch.quantile(values.detach().float().cpu(), torch.tensor([0.5, 0.9, 0.95]))
    print("{} mean/median/p90/p95/max: {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}".format(
        name, values.mean().item(), q[0].item(), q[1].item(), q[2].item(), values.max().item()
    ))
    return q


def candidate_original_paths(root, image_name):
    stem, _ = os.path.splitext(image_name)
    base = os.path.basename(stem)
    names = [image_name]
    names.extend(stem + ext for ext in EXTS)
    names.extend(base + ext for ext in EXTS)
    return [os.path.join(root, name) for name in dict.fromkeys(names)]


def load_image_tensor(path, device="cuda"):
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    image = image.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
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


def main():
    args, lp, pp = parse_args()
    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    xyz = gaussians.get_xyz.detach()
    bbox_min = xyz.min(dim=0).values
    bbox_max = xyz.max(dim=0).values
    bbox_diag = (bbox_max - bbox_min).norm().clamp_min(1e-8)
    scale_norm = gaussians.get_scaling.detach().norm(dim=-1)
    opacity = gaussians.get_opacity.detach().flatten()

    print("loaded iteration:", scene.loaded_iter)
    print("number of Gaussians:", xyz.shape[0])
    print("xyz bbox min:", bbox_min.cpu().tolist())
    print("xyz bbox max:", bbox_max.cpu().tolist())
    print("bbox diagonal: {:.8f}".format(bbox_diag.item()))
    scale_q = quantiles("scale norm", scale_norm)
    opacity_q = quantiles("opacity", opacity)

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
            original = load_image_tensor(original_path)
            source = render(cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)["render"].clamp(0, 1)
            original = resize_like(original, source)
            diff = (original[:3] - source).abs()
            l1 = diff.mean().item()
            iou = mask_iou(image_to_mask(original), image_to_mask(source)).item()
            per_view.append((cam.image_name, l1, iou))
            side_by_side([original[:3], source, heatmap(diff)]).save(
                os.path.join(args.out_dir, "{:02d}_{}_glb_source_diff.png".format(len(per_view), cam.image_name))
            )
            print("{} L1={:.6f} maskIoU={:.6f}".format(cam.image_name, l1, iou))
            if len(per_view) >= args.max_views:
                break

    avg_l1 = float(np.mean([x[1] for x in per_view])) if per_view else float("inf")
    avg_iou = float(np.mean([x[2] for x in per_view])) if per_view else 0.0
    scale_p95_ratio = scale_q[2].item() / bbox_diag.item()
    print("average image L1: {:.6f}".format(avg_l1))
    print("average foreground mask IoU: {:.6f}".format(avg_iou))
    print("p95 scale / bbox diagonal: {:.6f}".format(scale_p95_ratio))

    fail_reasons = []
    if xyz.shape[0] < 20000:
        fail_reasons.append("Gaussian count is low for a detailed object (<20k).")
    if scale_p95_ratio > 0.03:
        fail_reasons.append("p95 scale is large relative to bbox; render may be blurry.")
    if avg_iou < 0.75:
        fail_reasons.append("foreground mask IoU is low; silhouette may be misaligned.")
    if avg_l1 > 0.12:
        fail_reasons.append("average GLB-vs-source L1 is high.")
    if not per_view:
        fail_reasons.append("no matched views were found.")

    print("Final recommendation:")
    if fail_reasons:
        print("FAIL")
        for reason in fail_reasons:
            print("- " + reason)
        print("Fix source 3DGS before Stage 1/Stage 2.")
    else:
        print("PASS")
        print("Source may be good enough for xyz-only Stage 1, subject to visual inspection of saved panels.")


if __name__ == "__main__":
    main()
