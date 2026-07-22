#!/usr/bin/env python3
"""Evaluate extracted image observations against hidden synthetic projections.

This evaluator is separate from extraction and is the only component here that
may load a synthetic ground-truth delta.
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams
from correspondence.gaussian_visibility import project_points
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.observation_evaluation import endpoint_metrics, confidence_calibration
from correspondence.schema import ObservationBundle
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_path", required=True)
    p.add_argument("--gt_delta_path", required=True)
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--load_iteration", type=int, default=30000)
    a = p.parse_args()
    bundle = ObservationBundle.load(a.bundle_path, device="cpu")
    if bundle.observation_mode != "observed_2d" or bundle.target_xyz is not None:
        raise ValueError("evaluation requires an observed_2d bundle with target_xyz=None")
    gt = torch.load(a.gt_delta_path, map_location="cpu")
    target_xyz = gt["source_xyz"].float() + gt["d_xyz"].float()
    if torch.cuda.is_available():
        parser = argparse.ArgumentParser(add_help=False)
        lp = ModelParams(parser)
        args = parser.parse_args(["-s", a.source_path, "--model_path", a.model_path])
        safe_state(True)
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(lp.extract(args), gaussians, load_iteration=a.load_iteration, shuffle=False)
        cameras = {c.image_name: c for c in scene.getTrainCameras()}
    else:
        _, cpu_cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
        cameras = {c.image_name: c for c in cpu_cameras}
        print("WARNING: CUDA unavailable; evaluating with CPU PLY/transform cameras")
    oracle = []
    for name in bundle.camera_names:
        cam = cameras[name]
        target_device = cam.full_proj_transform.device
        xy, _, _ = project_points(target_xyz.to(target_device), cam.full_proj_transform,
                                  cam.image_width, cam.image_height)
        oracle.append(xy.cpu())
    oracle = torch.stack(oracle)
    valid = bundle.visibility_2d
    metrics = endpoint_metrics(bundle.target_xy, oracle, valid)
    error = torch.linalg.vector_norm(bundle.target_xy - oracle, dim=-1)
    confidence = bundle.confidence_2d if bundle.confidence_2d is not None else valid.float()
    flat_valid = valid & (confidence > 0)
    metrics.update({
        "visible_gaussian_coverage": float((bundle.support_count_2d > 0).float().mean()),
        "per_view_coverage": [float(x.float().mean()) for x in valid],
        "confidence_calibration": confidence_calibration(confidence, error, flat_valid),
        "mean_cycle_error": [x.get("mean_cycle_error") for x in bundle.metadata.get("per_view", [])],
        "matcher": bundle.metadata.get("matcher"),
        "visibility_method": bundle.metadata.get("visibility_method"),
        "observation_mode": bundle.observation_mode,
        "target_xyz_used_by_evaluator_only": True,
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
