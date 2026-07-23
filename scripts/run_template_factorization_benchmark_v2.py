#!/usr/bin/env python3
"""Stress-test shared style extraction under stochastic target templates.

The primary estimators do not receive nuisance coefficients. Oracle-label and
weak-label regressions are reported separately as upper bounds/stress tests.
Oracle target projections are used only to build observed_2d bundles and the
hidden teacher is loaded only by the evaluator.
"""
import argparse
import gc
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import recover_xyz_graph_coupled
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from stage1.template_factorization import delta_metrics, factorize_candidates


def normalize_active(x, active, target_norm, ratio):
    x = x.clone(); x[~active] = 0
    return x * (target_norm * ratio / x[active].norm().clamp_min(1e-8))


def make_bases(source, active, shared, seed, ratio):
    g = torch.Generator().manual_seed(seed)
    center = source[active].mean(0)
    q = source - center
    scale = q[active].abs().amax(0).clamp_min(1e-6)
    u = q / scale
    # Independent spatial fields: no compact=-radial duplication and no
    # hard-coded semantic labels. They are smooth nuisance directions.
    raw = [
        u,
        torch.stack([u[:, 0], 0.4 * u[:, 1], 0.2 * u[:, 2]], dim=1),
        torch.stack([torch.sin(u[:, 1] * 2.3), torch.cos(u[:, 0] * 1.7), u[:, 2] * u[:, 0]], dim=1),
        torch.stack([u[:, 0] * u[:, 1], u[:, 1] * u[:, 2], u[:, 2] * u[:, 0]], dim=1),
        torch.randn(source.shape, generator=g) * 0.25 + u * 0.15,
        torch.stack([torch.sin(q[:, 0] * 5.0), torch.sin(q[:, 1] * 4.0), torch.cos(q[:, 2] * 3.0)], dim=1),
    ]
    style_norm = shared[active].norm()
    bases = torch.stack([normalize_active(x, active, style_norm, ratio) for x in raw], 0)
    return bases


def make_candidates(source, shared, foreground, active, r, seed, ratio, noise=True):
    g = torch.Generator().manual_seed(seed)
    bases = make_bases(source, active, shared, seed + 101, ratio)
    coeff = torch.randn((r, bases.shape[0]), generator=g)
    nuisance = torch.einsum("rk,kni->rni", coeff, bases)
    candidates = shared[None] + nuisance
    if noise:
        # Recovery-like residual is small and zero outside the active region.
        eps = torch.randn(candidates.shape, generator=g) * float(shared[active].norm() / (active.sum().float().sqrt() * 40.0))
        candidates = candidates + eps * active[:, None].float()
    candidates[:, ~foreground] = 0
    return candidates, coeff, bases


def one_metrics(est, shared, foreground, active):
    out = delta_metrics(est, shared, active_mask=active, foreground_mask=foreground)
    out["nuisance_leakage"] = float((est[active] - shared[active]).square().sum() / shared[active].square().sum().clamp_min(1e-8))
    return out


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        for method, metrics in row["methods"].items():
            grouped[(row["templates"], row["ratio"], method)].append(metrics)
    summary = {}
    for key, values in grouped.items():
        summary["R{}_ratio{}_{}".format(key[0], key[1], key[2])] = {
            metric: {"mean": float(torch.tensor([v[metric] for v in values]).mean()),
                     "std": float(torch.tensor([v[metric] for v in values]).std(unbiased=False))}
            for metric in ("active_cosine", "energy_ratio", "explained_variance", "nuisance_leakage")
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt")
    p.add_argument("--output_dir", default="output/elephant_source_graphdeco/template_factorization_benchmark_v2")
    p.add_argument("--source_path", default="assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset")
    p.add_argument("--model_path", default="output/elephant_source_graphdeco")
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--run_recovered", action="store_true")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    teacher = torch.load(a.teacher, map_location="cpu")
    source, shared = teacher["source_xyz"].float(), teacher["d_xyz"].float()
    foreground = teacher["foreground_mask"].bool()
    active = teacher.get("synthetic_region_mask", foreground).bool()
    records = []
    seeds = [11, 29, 47]
    for ratio in (0.25, 0.5, 1.0):
        for r in (3, 5, 8):
            for seed in seeds:
                candidates, labels, _ = make_candidates(source, shared, foreground, active, r, seed, ratio)
                confidence = torch.ones((r, source.shape[0]))
                methods = factorize_candidates(candidates, confidence)
                # Weak labels are noisy, partly missing, discretized, and one
                # column intentionally corrupted; exact labels are upper bound.
                label_dim = max(1, min(labels.shape[1], r - 1))
                oracle_labels = labels[:, :label_dim]
                weak = oracle_labels + torch.randn(oracle_labels.shape, generator=torch.Generator().manual_seed(seed + 909)) * 0.35
                weak[:, 0] = torch.round(weak[:, 0] * 2.0) / 2.0
                if weak.shape[1] > 1:
                    weak[:, 1] = 0.0
                oracle_shared = factorize_candidates(candidates, confidence, oracle_labels)["nuisance_regression"]
                weak_shared = factorize_candidates(candidates, confidence, weak)["nuisance_regression"]
                methods["weak_label_regression"] = weak_shared
                methods["oracle_label_regression"] = oracle_shared
                record = {"templates": r, "ratio": ratio, "seed": seed, "label_dimension": label_dim,
                          "finite_sample_nuisance_mean": labels.mean(0).tolist(), "methods": {}}
                for name, value in methods.items():
                    if torch.is_tensor(value) and value.ndim == 2:
                        record["methods"][name] = one_metrics(value, shared, foreground, active)
                records.append(record)
    # Explicit non-identifiability test: all samples share a systematic nuisance.
    biased, _, _ = make_candidates(source, shared, foreground, active, 5, 101, 0.5)
    bias = make_bases(source, active, shared, 1001, 0.35)[0]
    biased = biased + bias[None]
    biased[:, ~foreground] = 0
    biased_methods = factorize_candidates(biased)
    biased_report = {name: one_metrics(value, shared, foreground, active)
                     for name, value in biased_methods.items() if torch.is_tensor(value) and value.ndim == 2}
    result = {"benchmark": "v2_stochastic_template_factorization", "teacher": a.teacher,
              "seeds": seeds, "ratios": [0.25, 0.5, 1.0], "template_counts": [3, 5, 8],
              "primary_methods": ["geometric_median", "robust_shared", "mean", "median"],
              "oracle_label_is_upper_bound": True, "summary": summarize(records),
              "records": records, "biased_nuisance": {"metrics": biased_report,
              "interpretation": "systematic target-template bias is not identifiable from one source without a prior, labels, or multiple sources"}}

    if a.run_recovered:
        source_loaded, cameras = load_cpu_source_and_cameras(a.source_path, a.model_path, a.load_iteration)
        base_bundle = ObservationBundle.load("output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt", device="cpu")
        selected = {c.image_name: c for c in cameras}
        selected = [selected[name] for name in base_bundle.camera_names]
        recovered_rows = []
        candidates, labels, _ = make_candidates(source, shared, foreground, active, 5, 29, 0.5)
        vis = base_bundle.visibility_2d.clone()
        for i in range(candidates.shape[0]):
            target_xy = torch.stack([project_points(source_loaded + candidates[i], c.full_proj_transform, c.image_width, c.image_height)[0] for c in selected])
            bundle = ObservationBundle(source_xyz=source_loaded, target_xy=target_xy, visibility_2d=vis,
                confidence_2d=vis.float(), support_count_2d=vis.sum(0), camera_names=base_bundle.camera_names,
                metadata={"oracle_target_projection": True, "target_xyz_in_optimizer_input": False}, observation_mode="observed_2d")
            path = os.path.join(a.output_dir, "recovered_bundle_{:02d}.pt".format(i)); bundle.save(path)
            rec = recover_xyz_graph_coupled(source_loaded, selected, target_xy, vis, vis.float(), iterations=3, graph_lambda=0.01, min_support=2)
            recovered_rows.append(rec["d_xyz"])
            del rec, bundle, target_xy
            gc.collect()
        recovered = torch.stack(recovered_rows)
        recovered_methods = factorize_candidates(recovered)
        result["recovered_oracle_candidates"] = {name: one_metrics(value, shared, foreground, active)
            for name, value in recovered_methods.items() if torch.is_tensor(value) and value.ndim == 2}
        result["recovered_candidate_metadata"] = {"count": 5, "observation_mode": "observed_2d", "target_xyz_in_optimizer_input": False}
    with open(os.path.join(a.output_dir, "benchmark_v2_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"summary": result["summary"], "biased_nuisance": result["biased_nuisance"],
                      "recovered_oracle_candidates": result.get("recovered_oracle_candidates")}, indent=2))


if __name__ == "__main__":
    main()
