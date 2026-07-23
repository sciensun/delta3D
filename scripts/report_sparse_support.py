#!/usr/bin/env python3
"""Report track support without running deformation recovery."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.schema import ObservationBundle
from correspondence.sparse_sampling import track_dropout, fixed_views_per_track, observability_report, pairwise_baseline_scores

def main():
    out='output/elephant_source_graphdeco/sparse_observation_benchmark'
    _, cameras=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000)
    b=ObservationBundle.load('output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt')
    payloads=[]; baseline=pairwise_baseline_scores([c for c in cameras if c.image_name in b.camera_names]);
    for teacher in ('body_roundness','ear_expansion','trunk_bending'):
        p=torch.load(f'output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher}.pt'); fg=p['foreground_mask'].bool(); active=p.get('synthetic_region_mask',fg).bool()
        for mode in ('all_valid_views_track_dropout','exact_k_views_per_track','baseline_maximized_k_views','random_k_views'):
            for frac in (.1,.2,.4,.6):
                for seed in (11,29,47,71,97):
                    if mode=='all_valid_views_track_dropout': vis,sel=track_dropout(b.visibility_2d,frac,fg,seed)
                    else:
                        k = 2 if mode in ('baseline_maximized_k_views','random_k_views') else 3
                        vis,sel=fixed_views_per_track(b.visibility_2d,frac,k,fg,seed,baseline_scores=baseline if mode=='baseline_maximized_k_views' else None)
                    r=observability_report(vis,fg,active); r.pop('track_counts',None)
                    payloads.append({'teacher':teacher,'mode':mode,'fraction':frac,'views_per_track':k if mode!='all_valid_views_track_dropout' else None,'seed':seed,'selected_tracks':int(sel.sum()),'report':r})
    path=os.path.join(out,'sparse_support_report.json');json.dump({'records':payloads,'definition':'fraction is Gaussian-track retention for track modes; independent_view_dropout remains per-view retention'},open(path,'w'),indent=2);print(path)
if __name__=='__main__':main()
