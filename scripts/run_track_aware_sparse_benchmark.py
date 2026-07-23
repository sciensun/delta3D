#!/usr/bin/env python3
"""Separate track observability from dense completion on synthetic teachers."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import build_geometry_cache, recover_xyz_graph_coupled_cached
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from correspondence.sparse_sampling import (track_dropout, fixed_views_per_track,
                                             observability_report)
from stage1.template_factorization import delta_metrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--teachers', nargs='+', default=['body_roundness','ear_expansion','trunk_bending']); ap.add_argument('--fractions', nargs='+', type=float, default=[.1,.2,.4,.6]); ap.add_argument('--seeds', nargs='+', type=int, default=[11,29,47,71,97]); ap.add_argument('--views_per_track', type=int, default=2); args=ap.parse_args()
    out = "output/elephant_source_graphdeco/sparse_observation_benchmark"
    source, cameras = load_cpu_source_and_cameras("assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset", "output/elephant_source_graphdeco", 30000)
    base = ObservationBundle.load("output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt")
    cams = [{c.image_name: c for c in cameras}[n] for n in base.camera_names]
    cache = build_geometry_cache(source, cams, knn=8); records=[]
    for teacher_name in args.teachers:
        p=torch.load(f"output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher_name}.pt")
        gt=p["d_xyz"].float(); fg=p["foreground_mask"].bool(); active=p.get("synthetic_region_mask",fg).bool()
        xy=torch.stack([project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)[0] for c in cams])
        for mode in ("track_dropout", "fixed_views_per_track"):
            for fraction in args.fractions:
                for seed in args.seeds:
                    if mode == "track_dropout":
                        vis, selected = track_dropout(base.visibility_2d, fraction, fg, seed)
                    else:
                        vis, selected = fixed_views_per_track(base.visibility_2d, fraction, args.views_per_track, fg, seed)
                    obs = vis.any(0); supports = vis.sum(0); anchors = fg & (supports >= 2)
                    started=time.perf_counter()
                    rec=recover_xyz_graph_coupled_cached(cache,xy,vis,vis.float(),iterations=12,graph_lambda=.01,foreground_mask=fg,jacobian_refresh=1)
                    report=delta_metrics(rec["d_xyz"],gt,active_mask=active,foreground_mask=fg)
                    report["anchor"] = delta_metrics(rec["d_xyz"],gt,active_mask=active & anchors,foreground_mask=fg)["active"]
                    report["unobserved"] = delta_metrics(rec["d_xyz"],gt,active_mask=active & ~obs,foreground_mask=fg)["active"]
                    report["support"] = observability_report(vis,fg,active)
                    records.append({"teacher":teacher_name,"mode":mode,"fraction":fraction,"seed":seed,
                        "selected_tracks":int(selected.sum()),"observed_tracks":int(obs.sum()),
                        "triangulatable_tracks":int(anchors.sum()),"report":report,
                        "runtime_seconds":time.perf_counter()-started})
    path=os.path.join(out,"track_aware_sparse_summary.json")
    existing=[]
    if os.path.exists(path):
        try: existing=json.load(open(path)).get("records", [])
        except Exception: existing=[]
    # Deterministic one-seed invocations can append, making the full five-seed
    # run robust to a long CPU process being interrupted.
    merged=existing + records
    payload={"benchmark":"track_aware_sparse_observability","records":merged,
             "modes":["independent_view_dropout","track_dropout","fixed_views_per_track"],
             "target_xyz_in_optimizer_input":False,"d_scaling_disabled":True,
             "primary_gate":"track fraction with >=2 valid views and adequate baseline"}
    json.dump(payload,open(path,"w"),indent=2)
    print(json.dumps({"records_added":len(records),"records_total":len(merged),"output":path},indent=2))

if __name__ == "__main__": main()
