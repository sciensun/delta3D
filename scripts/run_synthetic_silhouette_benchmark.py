#!/usr/bin/env python3
"""Synthetic silhouette controls using a depth/scale-aware splat fallback.

The CPU environment does not expose per-Gaussian renderer IDs, so this script
uses foreground-filtered, depth-sorted variable-radius alpha splats.  It is
explicitly marked approximate; it does not pass target XYZ to recovery.
"""
import json, os, sys
import cv2
import numpy as np
import torch
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import recover_xyz_from_observations
from correspondence.gaussian_visibility import project_points, camera_space_depth
from correspondence.silhouette import sample_silhouette_observation, signed_distance_and_gradient
from stage1.template_factorization import delta_metrics


def splat_mask(xy, depth, foreground, width, height):
    """Approximate alpha rasterization with depth order and variable footprint."""
    # Work on a quarter-resolution coverage buffer; the final mask is resized
    # back to the camera canvas for SDF sampling. This keeps the CPU fallback
    # practical while retaining depth ordering and variable footprints.
    scale = 4.0; work_w, work_h = max(1, width//4), max(1, height//4)
    xy = np.asarray(xy, np.float32) / scale; depth = np.asarray(depth, np.float32)
    fg = np.asarray(foreground).astype(bool)
    valid = fg & np.isfinite(depth) & (depth > 0)
    valid &= (xy[:, 0] >= -8) & (xy[:, 0] < width + 8)
    valid &= (xy[:, 1] >= -8) & (xy[:, 1] < height + 8)
    # A local projected-spacing proxy gives larger footprints in sparse areas.
    ix = np.clip(np.rint(xy[:, 0] / 8).astype(int), 0, max(0, width // 8))
    iy = np.clip(np.rint(xy[:, 1] / 8).astype(int), 0, max(0, height // 8))
    bins = {}
    for i in np.flatnonzero(valid): bins.setdefault((int(ix[i]), int(iy[i])), []).append(int(i))
    mask = np.zeros((work_h, work_w), np.float32)
    # Far-to-near compositing makes front Gaussians dominate.
    order = np.flatnonzero(valid)[np.argsort(depth[valid])[::-1]]
    # Keep the fallback bounded on the 44k-Gaussian bank; masks are silhouette
    # observations, not a per-Gaussian renderer replacement.
    if len(order) > 12000:
        order = order[np.linspace(0, len(order)-1, 12000).astype(int)]
    for i in order:
        key = (int(ix[i]), int(iy[i])); count = len(bins.get(key, ()))
        radius = float(np.clip(2.0 + 1.5 / np.sqrt(max(1, count)), 2.0, 4.0))
        alpha = float(np.clip(0.55 + 0.15 / max(depth[i], 1e-3), 0.35, 0.85))
        layer = np.zeros_like(mask)
        cv2.circle(layer, (int(round(xy[i,0])), int(round(xy[i,1]))), int(round(radius)), 1.0, -1)
        mask = mask + (1.0 - mask) * alpha * layer
    return cv2.resize((mask > 0.05).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def boundary_metrics(pred, target):
    pred = np.asarray(pred).astype(bool); target = np.asarray(target).astype(bool)
    def boundary(m):
        u = m.astype(np.uint8); er = cv2.erode(u, np.ones((3,3),np.uint8)); return (u-er).astype(bool)
    pb, tb = boundary(pred), boundary(target)
    dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_t = cv2.distanceTransform((~tb).astype(np.uint8), cv2.DIST_L2, 3)
    tp = float((pb & tb).sum()); precision = tp / max(1, float(pb.sum())); recall = tp / max(1, float(tb.sum()))
    return {'mask_iou': float((pred&target).sum()/max(1,(pred|target).sum())),
            'boundary_precision':precision,'boundary_recall':recall,
            'boundary_f1':2*precision*recall/max(1e-8,precision+recall),
            'symmetric_boundary_chamfer':float((dt_t[pb].mean() if pb.any() else 0)+(dt_p[tb].mean() if tb.any() else 0))/2}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--teachers', nargs='+', default=['body_roundness','ear_expansion','trunk_bending']); ap.add_argument('--modes', nargs='+', default=['correct_target','source_mask_null','shuffled_view']); args=ap.parse_args()
    source,cameras=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000)
    b=__import__('correspondence.schema',fromlist=['ObservationBundle']).ObservationBundle.load('output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt')
    cams=[{c.image_name:c for c in cameras}[n] for n in b.camera_names]; records=[]; root='output/elephant_source_graphdeco/sparse_observation_benchmark/silhouette_assets'
    for teacher in args.teachers:
        p=torch.load(f'output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher}.pt'); gt=p['d_xyz'].float(); fg=p['foreground_mask'].bool(); active=p.get('synthetic_region_mask',fg).bool(); src_xy=[]; tgt_xy=[]; src_masks=[]; tgt_masks=[]; sil={k:[] for k in ('valid','target_gradient','target_signed_distance')}
        teacher_root=os.path.join(root,teacher); os.makedirs(teacher_root,exist_ok=True)
        for vi,c in enumerate(cams):
            sxy,_,_=project_points(source,c.full_proj_transform,c.image_width,c.image_height); txy,_,_=project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)
            sd=camera_space_depth(source,c.world_view_transform).numpy(); td=camera_space_depth(source+gt,c.world_view_transform).numpy()
            sm=splat_mask(sxy.numpy(),sd,fg.numpy(),c.image_width,c.image_height); tm=splat_mask(txy.numpy(),td,fg.numpy(),c.image_width,c.image_height)
            obs=sample_silhouette_observation(sm,tm,sxy.numpy(),radius=4); src_xy.append(sxy); tgt_xy.append(txy); src_masks.append(sm); tgt_masks.append(tm)
            sdf,grad=signed_distance_and_gradient(tm)
            for k in sil: sil[k].append(torch.from_numpy(obs[k]))
            np.savez_compressed(os.path.join(teacher_root,f'{vi:02d}.npz'),source_mask=sm,target_mask=tm,target_sdf=sdf,target_gradient=grad)
        sil_t={k:torch.stack(v) for k,v in sil.items()}; zeros=torch.zeros_like(torch.stack(tgt_xy)); vis=torch.zeros((len(cams),len(source)),dtype=torch.bool)
        sdf_maps=[np.load(os.path.join(teacher_root,f'{vi:02d}.npz'))['target_sdf'] for vi in range(len(cams))]; grad_maps=[np.load(os.path.join(teacher_root,f'{vi:02d}.npz'))['target_gradient'] for vi in range(len(cams))]
        def run_sil(label, maps):
            gs=[]; ss=[]
            for m in maps:
                sdf,g=signed_distance_and_gradient(m); gs.append(g); ss.append(sdf)
            o={'valid':sil_t['valid'],'target_gradient':sil_t['target_gradient'],'target_signed_distance':sil_t['target_signed_distance'],'target_sdf_maps':ss,'target_gradient_maps':gs}
            rr=recover_xyz_from_observations(source,cams,zeros,vis,iterations=8,min_support=0,propagate=False,silhouette_observations=o,silhouette_weight=1.0,dynamic_silhouette=True,foreground_mask=fg)
            pred_masks=[]; post=[]; sdf_res=[]
            for vi,c in enumerate(cams):
                pxy,_,_=project_points(source+rr['d_xyz'],c.full_proj_transform,c.image_width,c.image_height)
                pd=camera_space_depth(source+rr['d_xyz'],c.world_view_transform).numpy()
                pm=splat_mask(pxy.numpy(),pd,fg.numpy(),c.image_width,c.image_height); pred_masks.append(pm); post.append(boundary_metrics(pm,tgt_masks[vi]))
                xx=np.clip(np.rint(pxy[:,0].numpy()).astype(int),0,c.image_width-1); yy=np.clip(np.rint(pxy[:,1].numpy()).astype(int),0,c.image_height-1)
                sdf_res.append(float(np.abs(sdf_maps[vi][yy,xx]).mean()))
            return {'teacher':teacher,'silhouette_mode':label,'delta_metrics':delta_metrics(rr['d_xyz'],gt,active_mask=active,foreground_mask=fg)['active'],'background_energy':float(rr['d_xyz'][~fg].square().sum()),'d_scaling_max':0.0,'target_xyz_in_solver':False,'silhouette_generation':'foreground_filtered_depth_sorted_variable_radius_alpha_splat_cpu_fallback','sdf_assets':teacher_root,'pre_image_metrics':{k:float(np.mean([boundary_metrics(src_masks[i],tgt_masks[i])[k] for i in range(len(cams))])) for k in boundary_metrics(src_masks[0],tgt_masks[0])},'post_image_metrics':{k:float(np.mean([x[k] for x in post])) for k in post[0]},'post_target_sdf_abs_mean':float(np.mean(sdf_res))}
        candidates={'correct_target':tgt_masks,'source_mask_null':src_masks,'shuffled_view':tgt_masks[1:]+tgt_masks[:1]}
        records.extend([run_sil(label,candidates[label]) for label in args.modes])
        image=[boundary_metrics(src_masks[i],tgt_masks[i]) for i in range(len(cams))]
        records.append({'teacher':teacher,'silhouette_mode':'mask_observation_quality','image_metrics':{k:float(np.mean([x[k] for x in image])) for k in image[0]},'target_xyz_in_solver':False,'silhouette_generation':'foreground_filtered_depth_sorted_variable_radius_alpha_splat_cpu_fallback','sdf_assets':teacher_root})
    path='output/elephant_source_graphdeco/sparse_observation_benchmark/synthetic_silhouette_summary.json'
    previous=[]
    if os.path.exists(path):
        try: previous=json.load(open(path)).get('records',[])
        except Exception: previous=[]
    keys={(r.get('teacher'),r.get('silhouette_mode')) for r in records}
    merged=[r for r in previous if (r.get('teacher'),r.get('silhouette_mode')) not in keys] + records
    json.dump({'records':merged,'controls':['correct_target','source_mask_null','shuffled_view'],'records_are_approximate_cpu_fallback':True},open(path,'w'),indent=2);print(path)
if __name__=='__main__': main()
