#!/usr/bin/env python3
"""Extract image-derived observed_2d observations without oracle geometry."""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams
from correspondence.image_observations import extract_image_observations
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--source_image_root", required=True)
    p.add_argument("--target_image_root", required=True)
    p.add_argument("--output_bundle", required=True)
    p.add_argument("--foreground_mask_path")
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--view_indices", default=None, help="comma-separated train-camera indices")
    p.add_argument("--search_radius", type=int, default=2)
    p.add_argument("--min_confidence", type=float, default=0.15)
    p.add_argument("--max_cycle_error", type=float, default=3.0)
    p.add_argument("--brightness", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--blur_sigma", type=float, default=0.0)
    p.add_argument("--mask_erode", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260722)
    a = p.parse_args()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        parser = argparse.ArgumentParser(add_help=False)
        lp = ModelParams(parser)
        args = parser.parse_args(["-s", a.source_path, "--model_path", a.model_path])
        safe_state(True)
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(lp.extract(args), gaussians, load_iteration=a.load_iteration, shuffle=False)
        source_xyz, cameras = gaussians.get_xyz.detach(), scene.getTrainCameras()
    else:
        source_xyz, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
        print("WARNING: CUDA unavailable; using CPU PLY/transform visibility diagnostic")
    if a.view_indices:
        indices = [int(x) for x in a.view_indices.split(",") if x.strip()]
        cameras = [cameras[i] for i in indices]
    perturb = {"brightness": a.brightness, "contrast": a.contrast,
               "noise_std": a.noise_std, "blur_sigma": a.blur_sigma,
               "mask_erode": a.mask_erode}
    bundle = extract_image_observations(
        source_xyz, cameras, a.source_image_root,
        a.target_image_root, foreground_mask_path=a.foreground_mask_path,
        device="cuda" if use_cuda else "cpu", search_radius=a.search_radius,
        min_confidence=a.min_confidence, max_cycle_error=a.max_cycle_error,
        target_perturb=perturb, perturb_seed=a.seed,
    )
    bundle.metadata.update({"source_path": a.source_path, "model_path": a.model_path,
                            "load_iteration": a.load_iteration, "device": "cuda" if use_cuda else "cpu", "view_indices":
                            [int(x) for x in (a.view_indices.split(",") if a.view_indices else range(len(cameras)))]})
    bundle.save(a.output_bundle)
    with open(os.path.splitext(a.output_bundle)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2)
    print(json.dumps({"output_bundle": a.output_bundle, "views": len(cameras),
                      "gaussians": int(bundle.source_xyz.shape[0]),
                      "support": int((bundle.support_count_2d > 0).sum())}, indent=2))


if __name__ == "__main__":
    main()
