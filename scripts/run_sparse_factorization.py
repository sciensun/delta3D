#!/usr/bin/env python3
"""Factorize five recovered oracle candidates after sparse observation recovery."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import build_geometry_cache, recover_xyz_graph_coupled_cached
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from stage1.template_factorization import delta_metrics, structured_no_label_factorization
from scripts.run_template_factorization_benchmark_v3 import make_candidates


def main():
    out = "output/elephant_source_graphdeco/sparse_observation_benchmark"
    source, cameras = load_cpu_source_and_cameras(
        "assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset",
        "output/elephant_source_graphdeco", 30000)
    bundle = ObservationBundle.load(
        "output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt")
    cams = [{c.image_name: c for c in cameras}[n] for n in bundle.camera_names]
    cache = build_geometry_cache(source, cams, knn=8)
    payload = torch.load("output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt")
    shared = payload["d_xyz"].float(); fg = payload["foreground_mask"].bool(); active = payload["synthetic_region_mask"].bool()
    candidates, _, _, _, _ = make_candidates(source, shared, fg, active, 5, 515, 0.5, "mixed", cache["neighbors"])
    records = {}
    for coverage in (0.2, 0.4, 1.0):
        recovered = []
        gen = torch.Generator().manual_seed(90210 + int(coverage * 100))
        for candidate in candidates:
            xy = torch.stack([project_points(source + candidate, c.full_proj_transform, c.image_width, c.image_height)[0] for c in cams])
            vis = bundle.visibility_2d.clone() & (torch.rand(bundle.visibility_2d.shape, generator=gen) < coverage)
            rec = recover_xyz_graph_coupled_cached(cache, xy, vis, vis.float(), iterations=12,
                graph_lambda=0.01, foreground_mask=fg, jacobian_refresh=1)
            recovered.append(rec["d_xyz"])
        recovered = torch.stack(recovered)
        mean = recovered.mean(0)
        graph_mean = mean.clone()
        graph_mean = 0.8 * graph_mean + 0.2 * graph_mean[cache["neighbors"]].mean(1)
        model = structured_no_label_factorization(recovered, rank=2, iterations=8,
            neighbors=cache["neighbors"], foreground_mask=fg, confidence=None)
        records[str(coverage)] = {
            "single": delta_metrics(recovered[0], shared, active, foreground_mask=fg),
            "mean": delta_metrics(mean, shared, active, foreground_mask=fg),
            "mean_plus_graph": delta_metrics(graph_mean, shared, active, foreground_mask=fg),
            "structured_no_label": delta_metrics(model["shared"], shared, active, foreground_mask=fg),
            "d_scaling_max": 0.0,
            "background_energy": float(model["shared"][~fg].square().sum())}
    path = os.path.join(out, "sparse_recovered_factorization.json")
    with open(path, "w") as h: json.dump({"coverage": records, "target_xyz_in_optimizer_input": False}, h, indent=2)
    print(json.dumps({"output": path, "coverage": list(records)}, indent=2))

if __name__ == "__main__": main()
