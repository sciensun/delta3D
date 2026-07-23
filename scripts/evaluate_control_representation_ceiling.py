#!/usr/bin/env python3
"""Hidden-GT evaluator for the approximation ceiling of control bases."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.control_graph import farthest_point_indices, interpolation_weights
from stage1.template_factorization import delta_metrics
from correspondence.cpu_cameras import load_cpu_source_and_cameras

def main():
    source,_=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000);out=[]
    for teacher in ('body_roundness','ear_expansion','trunk_bending'):
        p=torch.load(f'output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_{teacher}.pt');gt=p['d_xyz'].float();fg=p['foreground_mask'].bool();active=p.get('synthetic_region_mask',fg).bool()
        for m in (32,64,128,256):
            controls=farthest_point_indices(source,fg,m,11);idx,w=interpolation_weights(source,controls,fg);B=torch.zeros((len(source),len(controls)));B.scatter_add_(1,idx,w);A=B[fg];normal=A.T@A+1e-6*torch.eye(len(controls));coef=torch.linalg.solve(normal,A.T@gt[fg]);pred=B@coef;pred[~fg]=0;met=delta_metrics(pred,gt,active_mask=active,foreground_mask=fg);out.append({'teacher':teacher,'control_count':m,'active':met['active'],'foreground':met['foreground'],'background_energy':float(pred[~fg].square().sum()),'d_scaling_max':0.0})
    path='output/elephant_source_graphdeco/sparse_observation_benchmark/control_representation_ceiling.json';json.dump({'records':out,'hidden_gt_used_only_in_evaluator':True},open(path,'w'),indent=2);print(path)
if __name__=='__main__':main()
