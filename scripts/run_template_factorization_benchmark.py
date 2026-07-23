#!/usr/bin/env python3
"""Controlled shared-style/template-nuisance benchmark.

All candidate deltas are generated from one fixed Gaussian bank. This is an
evaluation of factorization, not an image-matching or real style experiment.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1.template_factorization import factorize_candidates, delta_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt")
    p.add_argument("--output_dir", default="output/elephant_source_graphdeco/template_factorization_benchmark")
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    os.makedirs(os.path.join(a.output_dir, "variants"), exist_ok=True)
    teacher = torch.load(a.teacher, map_location="cpu")
    source = teacher["source_xyz"].float()
    shared = teacher["d_xyz"].float()
    active = teacher.get("synthetic_region_mask", teacher["foreground_mask"]).bool()
    center = source[active].mean(0)
    q = source - center
    scale = q.abs().amax(0).clamp_min(1e-6)
    radial = q / scale * active[:, None].float()
    length = torch.zeros_like(source); length[:, 0] = q[:, 0] / scale[0] * active.float()
    compact = -q / scale * active[:, None].float()
    # A spatially local nuisance proxy. It is deliberately not assigned a
    # semantic label: the benchmark tests nuisance separation, not part names.
    ear_proxy = ((q[:, 1].abs() > torch.quantile(q[active, 1].abs(), 0.72)).float()[:, None] * q / scale)
    ear_proxy *= active[:, None].float()
    # Moderate nuisance is intentionally comparable to, but not larger than,
    # the shared teacher. This makes single-template estimates informative.
    bases = torch.stack([radial, length, compact, ear_proxy], 0) * 0.04
    coeff = torch.tensor([
        [-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    ])
    nuisance = torch.einsum("rk,kni->rni", coeff, bases)
    candidates = shared[None] + nuisance
    candidates[:, ~active] = 0
    confidence = torch.ones((5, source.shape[0]))
    for i in range(5):
        payload = {"d_xyz": candidates[i], "d_scaling": torch.zeros_like(candidates[i]),
                   "source_xyz": source, "foreground_mask": teacher["foreground_mask"],
                   "synthetic_region_mask": active, "metadata": {"variant": i, "shared_style": "body_roundness",
                   "nuisance_coefficients": coeff[i].tolist(), "appearance_only_nuisance": i in (1, 3)}}
        torch.save(payload, os.path.join(a.output_dir, "variants", "variant_{:02d}.pt".format(i)))
    torch.save({"deltas": candidates, "shared_gt": shared, "nuisance_gt": nuisance,
                "confidence": confidence, "source_xyz": source, "active_mask": active,
                "foreground_mask": teacher["foreground_mask"], "coefficients": coeff,
                "metadata": {"teacher": a.teacher, "seed": a.seed, "d_scaling_disabled": True}},
               os.path.join(a.output_dir, "candidate_stack.pt"))

    methods = factorize_candidates(candidates, confidence, coeff)
    summary = {"teacher": a.teacher, "seed": a.seed, "template_count": 5,
               "active_count": int(active.sum()), "d_scaling_max": 0.0, "methods": {}}
    for name, estimate in methods.items():
        if not torch.is_tensor(estimate) or estimate.ndim != 2:
            continue
        metrics = delta_metrics(estimate, shared, active)
        metrics["nuisance_leakage"] = float((estimate - shared).norm() / shared.norm().clamp_min(1e-8))
        summary["methods"][name] = metrics
        torch.save({"d_xyz": estimate, "d_scaling": torch.zeros_like(estimate),
                    "source_xyz": source, "foreground_mask": teacher["foreground_mask"],
                    "synthetic_region_mask": active, "metadata": {"factorization": name,
                    "shared_style_teacher": a.teacher, "d_scaling_disabled": True}},
                   os.path.join(a.output_dir, "shared_{}.pt".format(name)))

    # Robustness: one large outlier, unequal confidence, and a missing region.
    outlier = candidates.clone()
    outlier[0] = shared + torch.randn_like(shared) * 0.04 * active[:, None].float()
    unequal = confidence.clone(); unequal[0] *= 0.15
    missing = confidence.clone(); missing[2, active] = 0
    robust_cases = {
        "one_outlier": factorize_candidates(outlier, confidence, coeff)["robust_shared"],
        "unequal_confidence": factorize_candidates(candidates, unequal, coeff)["robust_shared"],
        "missing_region": factorize_candidates(candidates, missing, coeff)["robust_shared"],
    }
    summary["robustness"] = {name: delta_metrics(value, shared, active) for name, value in robust_cases.items()}
    with open(os.path.join(a.output_dir, "factorization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
