#!/usr/bin/env python3
"""Generate synthetic silhouette observations from teacher projections only."""
import json, os, sys
import cv2
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import recover_xyz_from_observations
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from correspondence.silhouette import sample_silhouette_observation
from stage1.template_factorization import delta_metrics

def mask_from_xy(xy, width, height, radius=3):
    mask=np.zeros((height,width),np.uint8)
    pts=np.asarray(torch.round(xy).numpy(),np.int32)
    for x,y in pts:
        if 0<=x<width and 0<=y<height: cv2.circle(mask,(int(x),int(y)),radius,1,-1)
    return mask.astype(bool)

def main():
    source,cameras=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000);b=ObservationBundle.load('output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt');cams=[{c.image_name:c for c in cameras}[n] for n in b.camera_names];records=[]
    for teacher in ('body_roundness','ear_expansion','trunk_bending'):
        p=torch.load(f'output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher}.pt');gt=p['d_xyz'].float();fg=p['foreground_mask'].bool();active=p.get('synthetic_region_mask',fg).bool();source_xy=[];target_xy=[];sil={k:[] for k in ('valid','source_boundary_distance','source_gradient','target_signed_distance','target_gradient')}
        for c in cams:
            sxy=project_points(source,c.full_proj_transform,c.image_width,c.image_height)[0]; txy=project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)[0]; sm=mask_from_xy(sxy,c.image_width,c.image_height); tm=mask_from_xy(txy,c.image_width,c.image_height);o=sample_silhouette_observation(sm,tm,sxy.numpy(),radius=4);source_xy.append(sxy);target_xy.append(txy)
            for k in sil:sil[k].append(torch.from_numpy(o[k]))
        sil_t={k:torch.stack(v) for k,v in sil.items()}; zeros=torch.zeros_like(torch.stack(target_xy)); vis=torch.zeros((len(cams),len(source)),dtype=torch.bool); srcm=torch.stack([torch.from_numpy(mask_from_xy(x,cams[0].image_width,cams[0].image_height)) for x in source_xy])
        obs={'valid':sil_t['valid'],'target_gradient':sil_t['target_gradient'],'target_signed_distance':sil_t['target_signed_distance']}
        rec=recover_xyz_from_observations(source,cams,zeros,vis,iterations=8,min_support=0,propagate=False,silhouette_observations=obs,silhouette_weight=1.0)
        correct=delta_metrics(rec['d_xyz'],gt,active_mask=active,foreground_mask=fg)['active']
        null_obs={'valid':sil_t['valid'],'target_gradient':sil_t['target_gradient'],'target_signed_distance':sil_t['source_boundary_distance']}
        null=recover_xyz_from_observations(source,cams,zeros,vis,iterations=8,min_support=0,propagate=False,silhouette_observations=null_obs,silhouette_weight=1.0)
        null_m=delta_metrics(null['d_xyz'],gt,active_mask=active,foreground_mask=fg)['active'];records.append({'teacher':teacher,'tracks_only':'not_run_in_this_script','correct_silhouette':correct,'null_source_silhouette':null_m,'background_energy':float(rec['d_xyz'][~fg].square().sum()),'d_scaling_max':0.0,'target_xyz_in_solver':False})
    path='output/elephant_source_graphdeco/sparse_observation_benchmark/synthetic_silhouette_summary.json';json.dump({'records':records,'silhouette_generation':'projected Gaussian footprint masks; hidden teacher used only to generate target masks'},open(path,'w'),indent=2);print(path)
if __name__=='__main__':main()
