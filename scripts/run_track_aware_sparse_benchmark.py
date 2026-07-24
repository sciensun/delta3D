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
                                             observability_report, pairwise_baseline_scores)
from correspondence.benchmark_artifacts import upsert_records, validate_records
from stage1.template_factorization import delta_metrics
from correspondence.track_consensus import leave_one_view_out_consensus, subset_hypothesis_consensus


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--teachers', nargs='+', default=['body_roundness','ear_expansion','trunk_bending']); ap.add_argument('--modes', nargs='+', default=['track_dropout','exact_k_views_per_track','baseline_maximized_k_views']); ap.add_argument('--fractions', nargs='+', type=float, default=[.1,.2,.4,.6]); ap.add_argument('--seeds', nargs='+', type=int, default=[11,29,47,71,97]); ap.add_argument('--views_per_track', type=int, default=2); ap.add_argument('--noise_px', type=float, default=0.0); ap.add_argument('--outlier_rate', type=float, default=0.0); ap.add_argument('--confidence_mode', choices=['calibrated','missing','overconfident_outliers'], default='calibrated'); ap.add_argument('--robust_kernel', choices=['huber','tukey','clip','reject'], default='huber'); ap.add_argument('--reject_threshold', type=float, default=8.0); ap.add_argument('--track_consensus', action='store_true'); ap.add_argument('--consensus_method', choices=['loo','subset'], default='loo'); ap.add_argument('--fallback', choices=['drop_track','keep_best_two','downgrade_confidence'], default='drop_track'); args=ap.parse_args()
    out = "output/elephant_source_graphdeco/sparse_observation_benchmark"
    source, cameras = load_cpu_source_and_cameras("assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset", "output/elephant_source_graphdeco", 30000)
    base = ObservationBundle.load("output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt")
    cams = [{c.image_name: c for c in cameras}[n] for n in base.camera_names]
    cache = build_geometry_cache(source, cams, knn=8); records=[]
    baseline = pairwise_baseline_scores(cams)
    for teacher_name in args.teachers:
        p=torch.load(f"output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher_name}.pt")
        gt=p["d_xyz"].float(); fg=p["foreground_mask"].bool(); active=p.get("synthetic_region_mask",fg).bool()
        xy=torch.stack([project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)[0] for c in cams])
        for mode in args.modes:
            for fraction in args.fractions:
                for seed in args.seeds:
                    policy = None
                    if mode == "track_dropout":
                        vis, selected = track_dropout(base.visibility_2d, fraction, fg, seed)
                    else:
                        policy = baseline if mode == "baseline_maximized_k_views" else None
                        vis, selected = fixed_views_per_track(base.visibility_2d, fraction, args.views_per_track, fg, seed, baseline_scores=policy)
                    input_vis = vis.clone()
                    obs = vis.any(0); supports = vis.sum(0); anchors = fg & (supports >= 2)
                    gen = torch.Generator().manual_seed(seed + 7001)
                    observed_xy = xy.clone()
                    if args.noise_px > 0:
                        observed_xy = observed_xy + torch.randn(observed_xy.shape, generator=gen) * args.noise_px
                    outlier_mask = vis & (torch.rand(vis.shape, generator=gen) < args.outlier_rate)
                    if outlier_mask.any():
                        observed_xy[outlier_mask] = torch.rand((int(outlier_mask.sum()), 2), generator=gen) * 1024.0
                    consensus = None
                    if args.track_consensus:
                        if args.consensus_method == 'subset':
                            consensus = subset_hypothesis_consensus(cache['source_views'], cache['jacobians'], observed_xy, vis, residual_threshold=args.reject_threshold, fallback=args.fallback)
                        else:
                            consensus = leave_one_view_out_consensus(cache['source_views'], cache['jacobians'], observed_xy, vis, residual_threshold=args.reject_threshold, fallback=args.fallback)
                        vis = consensus['accepted_visibility']
                        obs = vis.any(0); supports = vis.sum(0); anchors = fg & (supports >= 2)
                    if consensus is None:
                        true_reject = torch.zeros_like(input_vis)
                        false_reject = torch.zeros_like(input_vis)
                    else:
                        true_reject = outlier_mask & input_vis & ~vis
                        false_reject = (~outlier_mask) & input_vis & ~vis
                    outlier_views = (outlier_mask & input_vis).sum().item()
                    clean_views = ((~outlier_mask) & input_vis).sum().item()
                    confidence = vis.float()
                    if consensus is not None and 'track_confidence' in consensus:
                        confidence = confidence * consensus['track_confidence'][None, :]
                    if args.confidence_mode == 'missing': confidence = None
                    if args.confidence_mode == 'overconfident_outliers': confidence[outlier_mask] = 10.0
                    started=time.perf_counter()
                    rec=recover_xyz_graph_coupled_cached(cache,observed_xy,vis,confidence,iterations=12,graph_lambda=.01,foreground_mask=fg,jacobian_refresh=1,robust_kernel=args.robust_kernel,reject_threshold=args.reject_threshold)
                    report=delta_metrics(rec["d_xyz"],gt,active_mask=active,foreground_mask=fg)
                    report["anchor"] = delta_metrics(rec["d_xyz"],gt,active_mask=active & anchors,foreground_mask=fg)["active"]
                    report["unobserved"] = delta_metrics(rec["d_xyz"],gt,active_mask=active & ~obs,foreground_mask=fg)["active"]
                    report["support"] = observability_report(vis,fg,active)
                    injected = outlier_mask.any(0) & input_vis.any(0)
                    retained_outlier = (outlier_mask & vis).any(0)
                    no_outlier = ~injected
                    all_removed = injected & ~retained_outlier
                    residual_outlier = injected & retained_outlier
                    report['track_categories']={
                        'no_injected_outlier': delta_metrics(rec['d_xyz'],gt,active_mask=active&no_outlier,foreground_mask=fg),
                        'all_injected_outliers_removed': delta_metrics(rec['d_xyz'],gt,active_mask=active&all_removed,foreground_mask=fg),
                        'accepted_outlier_remaining': delta_metrics(rec['d_xyz'],gt,active_mask=active&residual_outlier,foreground_mask=fg),
                        'unobserved': delta_metrics(rec['d_xyz'],gt,active_mask=active&~obs,foreground_mask=fg)}
                    report["rejected_fraction"] = rec["history"][-1].get("rejected_fraction", 0.0)
                    records.append({"teacher":teacher_name,"mode":mode,"fraction":fraction,"views_per_track":args.views_per_track,
                        "baseline_policy":"max_pairwise_center_baseline" if policy is not None else "random",
                        "seed":seed,"noise":args.noise_px,"outlier_rate":args.outlier_rate,"confidence_mode":args.confidence_mode,"robust_kernel":args.robust_kernel,"reject_threshold":args.reject_threshold,"track_consensus":args.track_consensus,"consensus_method":args.consensus_method,"fallback":args.fallback,"solver":"symmetric_graph_irls",
                        "control_count":None,
                        "selected_tracks":int(selected.sum()),"observed_tracks":int(obs.sum()),
                        "consensus_rejected_fraction":0.0 if consensus is None else float((input_vis & ~vis).float().sum()/input_vis.float().sum().clamp_min(1)),
                        "true_outlier_rejection_rate":0.0 if outlier_views == 0 else float(true_reject.sum().item()/outlier_views),
                        "clean_false_rejection_rate":0.0 if clean_views == 0 else float(false_reject.sum().item()/clean_views),
                        "accepted_view_histogram":{str(k):int((supports==k).sum()) for k in range(int(input_vis.shape[0])+1)},
                        "consensus_available_tracks":int((input_vis.sum(0)>=3).sum()),
                        "retained_tracks_with_outlier":int(residual_outlier.sum()),
                        "retained_clean_mass":float(((~outlier_mask)&vis).float().sum()),
                        "retained_outlier_mass":float((outlier_mask&vis).float().sum()),
                        "triangulatable_tracks":int(anchors.sum()),"report":report,
                        "runtime_seconds":time.perf_counter()-started})
    path=os.path.join(out,"track_aware_sparse_summary.json")
    existing=[]
    if os.path.exists(path):
        try: existing=json.load(open(path)).get("records", [])
        except Exception: existing=[]
    # Deterministic one-seed invocations can append, making the full five-seed
    # run robust to a long CPU process being interrupted.
    merged=upsert_records(existing, records)
    payload={"benchmark":"track_aware_sparse_observability","records":merged,
             "modes":["independent_view_dropout","track_dropout","fixed_views_per_track"],
             "target_xyz_in_optimizer_input":False,"d_scaling_disabled":True,
             "primary_gate":"track fraction with >=2 valid views and adequate baseline",
             "validation":validate_records(merged)}
    json.dump(payload,open(path,"w"),indent=2)
    print(json.dumps({"records_added":len(records),"records_total":len(merged),"output":path},indent=2))

if __name__ == "__main__": main()
