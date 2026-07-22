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
from correspondence.observation_evaluation import (confidence_calibration,
                                                    endpoint_metrics,
                                                    evaluate_view_observations)
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
    p.add_argument("--foreground_mask_path")
    p.add_argument("--load_iteration", type=int, default=30000)
    a = p.parse_args()
    bundle = ObservationBundle.load(a.bundle_path, device="cpu")
    if bundle.observation_mode != "observed_2d" or bundle.target_xyz is not None:
        raise ValueError("evaluation requires an observed_2d bundle with target_xyz=None")
    gt = torch.load(a.gt_delta_path, map_location="cpu")
    target_xyz = gt["source_xyz"].float() + gt["d_xyz"].float()
    foreground = torch.load(a.foreground_mask_path, map_location="cpu").bool().flatten() if a.foreground_mask_path else torch.ones(bundle.source_xyz.shape[0], dtype=torch.bool)
    active = gt.get("synthetic_region_mask", foreground).bool().flatten()
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
    source_xy, oracle = [], []
    for name in bundle.camera_names:
        cam = cameras[name]
        target_device = cam.full_proj_transform.device
        sx, _, _ = project_points(bundle.source_xyz.to(target_device), cam.full_proj_transform,
                                   cam.image_width, cam.image_height)
        xy, _, _ = project_points(target_xyz.to(target_device), cam.full_proj_transform,
                                  cam.image_width, cam.image_height)
        source_xy.append(sx.cpu())
        oracle.append(xy.cpu())
    source_xy = torch.stack(source_xy)
    oracle = torch.stack(oracle)
    valid = bundle.visibility_2d
    candidates = bundle.candidate_visibility_2d if bundle.candidate_visibility_2d is not None else valid
    metrics = endpoint_metrics(bundle.target_xy, oracle, valid)
    error = torch.linalg.vector_norm(bundle.target_xy - oracle, dim=-1)
    confidence = bundle.confidence_2d if bundle.confidence_2d is not None else valid.float()
    flat_valid = valid & (confidence > 0)
    per_view = [evaluate_view_observations(bundle.target_xy[i], oracle[i], valid[i], candidates[i],
                                           confidence[i], source_xy[i], foreground, active)
                for i in range(valid.shape[0])]
    aggregate = evaluate_view_observations(bundle.target_xy.reshape(-1, 2), oracle.reshape(-1, 2),
                                           valid.reshape(-1), candidates.reshape(-1), confidence.reshape(-1),
                                           source_xy.reshape(-1, 2), foreground.repeat(valid.shape[0]),
                                           active.repeat(valid.shape[0]))
    metrics.update({
        "all_gaussian_support_coverage": aggregate["all_gaussian_support_coverage"],
        "foreground_support_coverage": aggregate["foreground_support_coverage"],
        "active_region_coverage": aggregate["active_region_coverage"],
        "inactive_region_coverage": aggregate["inactive_region_coverage"],
        "accepted_match_recall": aggregate["accepted_match_recall"],
        "active_accepted_match_recall": aggregate["active_accepted_match_recall"],
        "active": aggregate["active"],
        "inactive": aggregate["inactive"],
        "active_stratified_displacement": aggregate["active_stratified_displacement"],
        "confidence_precision_coverage": aggregate["confidence_precision_coverage"],
        "zero_motion": aggregate["zero_motion"],
        "zero_motion_active": aggregate["zero_motion_active"],
        "per_view": per_view,
        "visible_candidate_count": aggregate["visible_candidate_count"],
        "accepted_match_count": aggregate["accepted_match_count"],
        "per_view_coverage": [float(x.float().mean()) for x in valid],
        "confidence_calibration": confidence_calibration(confidence, error, flat_valid),
        "mean_cycle_error": [x.get("mean_cycle_error") for x in bundle.metadata.get("per_view", [])],
        "matcher": bundle.metadata.get("matcher"),
        "visibility_method": bundle.metadata.get("visibility_method"),
        "active_mask_source": "hidden_synthetic_region_mask_evaluator_only",
        "observation_mode": bundle.observation_mode,
        "target_xyz_used_by_evaluator_only": True,
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)), exist_ok=True)
    with open(a.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
