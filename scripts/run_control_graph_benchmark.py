#!/usr/bin/env python3
"""Evaluate low-dimensional control-node translations on synthetic teachers."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import build_geometry_cache
from correspondence.control_graph import fit_control_translation
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from correspondence.sparse_sampling import track_dropout
from stage1.template_factorization import delta_metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teachers',nargs='+',default=['body_roundness','ear_expansion','trunk_bending']);ap.add_argument('--fractions',nargs='+',type=float,default=[.2,.4]);ap.add_argument('--control_counts',nargs='+',type=int,default=[32,64,128,256]);args=ap.parse_args()
    out='output/elephant_source_graphdeco/sparse_observation_benchmark'; source,cameras=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000); b=ObservationBundle.load('output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt'); cams=[{c.image_name:c for c in cameras}[n] for n in b.camera_names]; cache=build_geometry_cache(source,cams,8); records=[]
    for teacher in args.teachers:
        p=torch.load(f'output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher}.pt');gt=p['d_xyz'].float();fg=p['foreground_mask'].bool();active=p.get('synthetic_region_mask',fg).bool();xy=torch.stack([project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)[0] for c in cams])
        for frac in args.fractions:
            vis,_=track_dropout(b.visibility_2d,frac,fg,11)
            for m in args.control_counts:
                started=time.perf_counter(); r=fit_control_translation(cache,xy,vis,fg,control_count=m,iterations=2,seed=11); met=delta_metrics(r['d_xyz'],gt,active_mask=active,foreground_mask=fg);records.append({'teacher':teacher,'track_fraction':frac,'control_count':m,'active':met['active'],'background_energy':float(r['d_xyz'][~fg].square().sum()),'d_scaling_max':0.0,'seconds':time.perf_counter()-started})
    path=os.path.join(out,'control_graph_benchmark.json'); existing=[]
    if os.path.exists(path):
        try: existing=json.load(open(path)).get('records',[])
        except Exception: pass
    merged=existing+records
    json.dump({'records':merged,'target_xyz_in_optimizer_input':False},open(path,'w'),indent=2);print(json.dumps({'records_added':len(records),'records_total':len(merged),'output':path},indent=2))
if __name__=='__main__':main()
