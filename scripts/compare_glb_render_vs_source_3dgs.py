#!/usr/bin/env python3
"""Compare original GLB renders against trained source 3DGS renders."""

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
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(), lp, pp


def candidate_original_paths(root, image_name):
    stem, ext = os.path.splitext(image_name)
    names = [image_name]
    names.extend(stem + suffix for suffix in EXTS)
    if stem.startswith("train/"):
        names.extend(os.path.basename(stem) + suffix for suffix in EXTS)
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
            panel = side_by_side([original[:3], source, heatmap(diff)])
            panel.save(os.path.join(args.out_dir, "{:02d}_{}_glb_source_diff.png".format(len(per_view), cam.image_name)))
            print("{} L1={:.6f} maskIoU={:.6f}".format(cam.image_name, l1, iou))
            if len(per_view) >= args.max_views:
                break

    if not per_view:
        print("No matched views found. Check --original_render_root and source camera image_name values.")
        return

    avg_l1 = float(np.mean([x[1] for x in per_view]))
    avg_iou = float(np.mean([x[2] for x in per_view]))
    print("Average L1: {:.6f}".format(avg_l1))
    print("Average mask IoU: {:.6f}".format(avg_iou))
    print("Interpretation:")
    print("- If source 3DGS is much blurrier than GLB render, Stage 0/source reconstruction is the bottleneck.")
    print("- If source 3DGS is clear, then Stage 1 delta mining is the bottleneck.")


if __name__ == "__main__":
    main()
