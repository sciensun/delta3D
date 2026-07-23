#!/usr/bin/env python3
"""Compare local IRLS and the matched undirected PCG quadratic on one teacher."""
import json, os, sys, time, resource
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.cpu_cameras import load_cpu_source_and_cameras
from correspondence.cpu_recovery import (build_geometry_cache, recover_xyz_graph_coupled_cached,
                                           solve_symmetric_graph_block)
from correspondence.gaussian_visibility import project_points
from correspondence.schema import ObservationBundle
from correspondence.sparse_sampling import track_dropout
from stage1.template_factorization import delta_metrics

def main():
    source,cameras=load_cpu_source_and_cameras('assets/prepared/big_carved_wooden_elephant_sculpture/stage1_5_key8_dataset','output/elephant_source_graphdeco',30000);b=ObservationBundle.load('output/elephant_source_graphdeco/synthetic_observed_2d_benchmark/clean_8/observed_2d_bundle.pt');cams=[{c.image_name:c for c in cameras}[n] for n in b.camera_names];cache=build_geometry_cache(source,cams,8)
    p=torch.load('output/elephant_source_graphdeco/synthetic_known_delta/synthetic_delta_body_roundness.pt');gt=p['d_xyz'].float();fg=p['foreground_mask'].bool();active=p['synthetic_region_mask'].bool();xy=torch.stack([project_points(source+gt,c.full_proj_transform,c.image_width,c.image_height)[0] for c in cams]);vis,_=track_dropout(b.visibility_2d,.2,fg,11);w=vis.float();j=cache['jacobians'];normal=torch.einsum('vn,vnij,vnik->njk',w,j,j)+1e-3*torch.eye(3).expand(len(source),3,3);rhs=torch.einsum('vn,vnij,vni->nj',w,j,xy-cache['source_views']);started=time.perf_counter();pcg,info=solve_symmetric_graph_block(normal,rhs,cache['edge_index'],cache['edge_weights'],graph_lambda=.01,maxiter=100);pcg[~fg]=0;pcg_time=time.perf_counter()-started;started=time.perf_counter();local=recover_xyz_graph_coupled_cached(cache,xy,vis,w,iterations=100,graph_lambda=.01,foreground_mask=fg,jacobian_refresh=1);local_time=time.perf_counter()-started
    result={'local_irls_100':delta_metrics(local['d_xyz'],gt,active_mask=active,foreground_mask=fg)['active'],'pcg_linearized':delta_metrics(pcg,gt,active_mask=active,foreground_mask=fg)['active'],'pcg_info':info,'local_seconds':local_time,'pcg_seconds':pcg_time,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'edge_count':int(cache['edge_index'].shape[0]),'graph_diagnostics':cache['graph_diagnostics'],'target_xyz_in_solver':False}
    path='output/elephant_source_graphdeco/sparse_observation_benchmark/fullbank_graph_solver_comparison.json';json.dump(result,open(path,'w'),indent=2);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
