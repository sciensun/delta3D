#!/usr/bin/env python3
"""Calibrated stochastic factorization and full-bank recovered-candidate gate."""
import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import build_geometry_cache, recover_xyz_graph_coupled_cached
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from stage1.template_factorization import (delta_metrics, factorize_candidates,
                                            structured_no_label_factorization)


def smooth_basis(raw, xyz, neighbors):
    field = raw.clone()
    for _ in range(3):
        field = 0.65 * field + 0.35 * field[neighbors].mean(1)
    return field


def make_nuisance_bases(xyz, foreground, active, shared, seed, support_condition, neighbors):
    gen = torch.Generator().manual_seed(seed)
    q = xyz - xyz[foreground].mean(0)
    scale = q[foreground].abs().amax(0).clamp_min(1e-6)
    u = q / scale
    outside = foreground & ~active
    if support_condition == "inside":
        support = active
    elif support_condition == "outside":
        support = outside
    else:
        support = foreground
    raw = [
        torch.stack([u[:, 0], 0.5 * u[:, 1], 0.2 * u[:, 2]], 1),
        torch.stack([u[:, 0] * u[:, 1], u[:, 1] * u[:, 2], u[:, 2] * u[:, 0]], 1),
        torch.stack([torch.sin(u[:, 1] * 2.1), torch.cos(u[:, 0] * 1.7), u[:, 2]], 1),
        torch.stack([torch.sin(q[:, 0] * 4.0), torch.sin(q[:, 1] * 3.5), torch.cos(q[:, 2] * 3.0)], 1),
        smooth_basis(torch.randn(xyz.shape, generator=gen), xyz, neighbors),
        smooth_basis(torch.randn(xyz.shape, generator=gen), xyz, neighbors),
    ]
    bases = []
    style_norm = shared[active].norm()
    for idx, value in enumerate(raw):
        value = value * support[:, None].float()
        if support_condition == "mixed" and idx % 2 == 0:
            value = value * (0.55 * active[:, None].float() + 0.45 * outside[:, None].float())
        value = value - value[foreground].mean(0)
        value[~foreground] = 0
        value = value / value[foreground].norm().clamp_min(1e-8)
        bases.append(value * style_norm)
    bases = torch.stack(bases)
    # Decorrelate under the foreground inner product, then retain distinct
    # spatial support. QR here is over the flattened foreground fields.
    flat = bases[:, foreground].reshape(len(bases), -1).T
    qmat, _ = torch.linalg.qr(flat, mode="reduced")
    decorrelated = qmat.T.reshape(len(bases), int(foreground.sum()), 3)
    out = torch.zeros_like(bases)
    out[:, foreground] = decorrelated
    out = out * (bases[:, foreground].norm(dim=(1, 2)) / out[:, foreground].norm(dim=(1, 2)).clamp_min(1e-8))[:, None, None]
    return out


def make_candidates(xyz, shared, foreground, active, r, seed, total_ratio, support_condition, neighbors):
    gen = torch.Generator().manual_seed(seed)
    bases = make_nuisance_bases(xyz, foreground, active, shared, seed + 101, support_condition, neighbors)
    coefficients = torch.randn((r, len(bases)), generator=gen)
    nuisance = torch.einsum("rk,kni->rni", coefficients, bases)
    shared_norm = shared[foreground].norm()
    realized = nuisance[:, foreground].norm(dim=(1, 2)).clamp_min(1e-8)
    nuisance *= (shared_norm * total_ratio / realized)[:, None, None]
    noise = torch.randn(nuisance.shape, generator=gen) * (shared_norm / (foreground.sum().float().sqrt() * 80.0))
    noise *= foreground[None, :, None].float()
    candidates = shared[None] + nuisance + noise
    candidates[:, ~foreground] = 0
    return candidates, nuisance, coefficients, bases, noise


def metrics_for(value, shared, foreground, active):
    result = delta_metrics(value, shared, active_mask=active, foreground_mask=foreground)
    result["nuisance_leakage"] = float((value[active] - shared[active]).square().sum() / shared[active].square().sum().clamp_min(1e-8))
    return result


def method_metrics(candidates, shared, foreground, active, neighbors, labels=None, confidence=None, structured=True):
    methods = factorize_candidates(candidates, confidence=confidence, geometric_iterations=8, robust_iterations=3)
    if structured:
        model = structured_no_label_factorization(candidates, rank=min(2, candidates.shape[0] - 1),
                                                  iterations=12, neighbors=neighbors,
                                                  foreground_mask=foreground)
        methods["structured_no_label"] = model["shared"]
    if labels is not None:
        methods["oracle_label_regression"] = factorize_candidates(candidates, nuisance_features=labels)["nuisance_regression"]
    return {name: metrics_for(value, shared, foreground, active)
            for name, value in methods.items() if torch.is_tensor(value) and value.ndim == 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt")
    p.add_argument("--output_dir", default="output/elephant_source_graphdeco/template_factorization_benchmark_v3")
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--skip_recovered", action="store_true")
    a = p.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    teacher = torch.load(a.teacher, map_location="cpu")
    xyz, shared = teacher["source_xyz"].float(), teacher["d_xyz"].float()
    foreground = teacher["foreground_mask"].bool(); active = teacher.get("synthetic_region_mask", foreground).bool()
    neighbors = torch.from_numpy(__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(xyz.numpy()).query(xyz.numpy(), k=9)[1][:, 1:]).long()
    all_records = []
    for condition in ("inside", "outside", "mixed"):
        for ratio in (0.25, 0.5, 1.0):
            for r in (3, 5, 8):
                for seed in (11, 29, 47, 71, 97):
                    cand, nuisance, coeff, bases, noise = make_candidates(xyz, shared, foreground, active, r, seed, ratio, condition, neighbors)
                    do_structured = condition == "mixed" and ratio == 0.5 and r == 5
                    record = {"condition": condition, "templates": r, "total_ratio_target": ratio,
                              "seed": seed, "realized_nuisance_style_ratio": float(nuisance[:, foreground].norm() / (shared[foreground].norm() * r ** 0.5)),
                              "realized_nuisance_energy_style_energy": float(nuisance[:, foreground].square().sum() / r / shared[foreground].square().sum()),
                              "outside_nuisance_energy_style_energy": float(nuisance[:, foreground & ~active].square().sum() / r / shared[active].square().sum().clamp_min(1e-8)),
                              "noise_style_ratio": float(noise[:, foreground].norm() / shared[foreground].norm()),
                              "finite_nuisance_mean_norm": float(nuisance.mean(0)[foreground].norm()),
                              "basis_gram": torch.einsum("kni,lni->kl", bases[:, foreground], bases[:, foreground]).tolist(),
                              "basis_singular_values": torch.linalg.svdvals(bases[:, foreground].reshape(len(bases), -1)).tolist(),
                              "methods": method_metrics(cand, shared, foreground, active, neighbors, labels=None, structured=do_structured)}
                    all_records.append(record)
    # Actual robustness branches on one calibrated R=5 mixed condition.
    cand, nuisance, coeff, bases, noise = make_candidates(xyz, shared, foreground, active, 5, 131, 0.5, "mixed", neighbors)
    robust = {}
    outlier = cand.clone(); outlier[0] = shared + torch.randn_like(shared) * (shared[foreground].norm() / foreground.sum().float().sqrt() * 2.0) * foreground[:, None].float()
    robust["one_outlier"] = method_metrics(outlier, shared, foreground, active, neighbors)
    two = outlier.clone(); two[1] = shared + torch.randn_like(shared) * (shared[foreground].norm() / foreground.sum().float().sqrt() * 2.0) * foreground[:, None].float()
    robust["two_outliers"] = method_metrics(two, shared, foreground, active, neighbors)
    conf = torch.ones((5, len(xyz))); conf[0] = 0.2
    robust["unequal_confidence"] = method_metrics(cand, shared, foreground, active, neighbors, confidence=conf)
    missing = cand.clone(); active_indices = torch.nonzero(active).flatten(); missing[2, active_indices[::2]] = shared[active_indices[::2]]
    robust["missing_local_region"] = method_metrics(missing, shared, foreground, active, neighbors)
    invalid = cand.clone(); invalid[3, foreground] = 0
    robust["partially_invalid"] = method_metrics(invalid, shared, foreground, active, neighbors)
    robust["poor_recovered_candidate"] = method_metrics(torch.cat([cand[:4], outlier[:1]], 0), shared, foreground, active, neighbors)
    result = {"benchmark": "v3_calibrated_stochastic_factorization", "records": all_records,
              "robustness": robust, "primary_methods_use_no_labels": True,
              "d_scaling_max": 0.0, "background_energy": 0.0,
              "v1_reclassified": "controlled implementation sanity check"}

    if not a.skip_recovered:
        source_loaded, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
        base = ObservationBundle.load("output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt", device="cpu")
        selected = [{c.image_name: c for c in cameras}[name] for name in base.camera_names]
        cache = build_geometry_cache(source_loaded, selected, knn=8)
        recovered = []; bundle_dir = os.path.join(a.output_dir, "recovered_bundles"); os.makedirs(bundle_dir, exist_ok=True)
        cand, nuisance, coeff, bases, noise = make_candidates(xyz, shared, foreground, active, 5, 29, 0.5, "mixed", neighbors)
        timings = []
        for idx in range(5):
            target_xy = torch.stack([project_points(source_loaded + cand[idx], c.full_proj_transform, c.image_width, c.image_height)[0] for c in selected])
            bundle = ObservationBundle(source_xyz=source_loaded, target_xy=target_xy, visibility_2d=base.visibility_2d,
                confidence_2d=base.visibility_2d.float(), support_count_2d=base.visibility_2d.sum(0), camera_names=base.camera_names,
                metadata={"observation_mode": "observed_2d", "target_xyz_in_optimizer_input": False}, observation_mode="observed_2d")
            bundle.save(os.path.join(bundle_dir, "candidate_{:02d}.pt".format(idx)))
            rec = recover_xyz_graph_coupled_cached(cache, target_xy, base.visibility_2d, base.visibility_2d.float(), iterations=8, graph_lambda=0.01, min_support=2)
            recovered.append(rec["d_xyz"]); timings.append(rec["recovery_seconds"])
        recovered = torch.stack(recovered)
        methods = factorize_candidates(recovered)
        structured = structured_no_label_factorization(recovered, rank=2, iterations=12,
                                                       neighbors=cache["neighbors"], foreground_mask=foreground)
        methods["structured_no_label"] = structured["shared"]
        result["recovered_full_bank"] = {"metrics": {name: metrics_for(value, shared, foreground, active)
            for name, value in methods.items() if torch.is_tensor(value) and value.ndim == 2},
            "gaussians": int(len(source_loaded)), "views": len(selected), "graph_k": 8,
            "cache_build_seconds": cache["cache_build_seconds"], "per_candidate_seconds": timings,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "target_xyz_in_optimizer_input": False}
        torch.save({"deltas": recovered, "d_scaling": torch.zeros_like(recovered), "metadata": {"target_xyz_used": False}}, os.path.join(a.output_dir, "recovered_candidates.pt"))
    with open(os.path.join(a.output_dir, "benchmark_v3_summary.json"), "w") as handle: json.dump(result, handle, indent=2)
    print(json.dumps({"records": len(all_records), "robustness": list(robust), "recovered_full_bank": result.get("recovered_full_bank")}, indent=2))


if __name__ == "__main__": main()
