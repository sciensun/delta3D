#!/usr/bin/env python3
"""Validate fixed-bank identity and conservative geometry diagnostics."""
import argparse, json, os, sys
import torch
from plyfile import PlyData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.canonical_ordinary_model import CanonicalOrdinaryModel

def xyz_from_ply(path):
    v=PlyData.read(path)["vertex"].data
    return torch.stack([torch.from_numpy(v[n].copy()) for n in ("x","y","z")],1).float()

def knn_dist(x,k=8,limit=4096):
    ids=torch.arange(len(x));
    if len(ids)>limit: ids=torch.linspace(0,len(x)-1,limit).long()
    d=torch.cdist(x[ids],x); d[:,ids]=d[:,ids]+torch.eye(len(ids))*1e6
    return d.topk(min(k,x.shape[0]-1),largest=False).values

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source_ply",required=True); p.add_argument("--canonical_path",required=True); p.add_argument("--output_path",required=True); p.add_argument("--exported_ply",default=None); a=p.parse_args()
    source=xyz_from_ply(a.source_ply); x=torch.load(a.canonical_path,map_location="cpu"); model=CanonicalOrdinaryModel(x["source_xyz"],x["d_xyz"],x.get("d_scaling"),x.get("foreground_mask"),x.get("correspondence_confidence"),x.get("metadata"));
    if source.shape!=model.source_xyz.shape or not torch.allclose(source,model.source_xyz,atol=1e-5): raise ValueError("source ordering/xyz changed")
    before=knn_dist(source); after=knn_dist(model.canonical_xyz); ratio=after/(before+1e-8)
    delta_norm=model.d_xyz.norm(dim=-1)
    report={"same_count":True,"same_order":True,"num_gaussians":model.num_gaussians,"background_delta_zero":bool((model.d_xyz[~model.foreground_mask]==0).all()) if model.foreground_mask is not None else None,"d_scaling_zero":bool(torch.equal(model.d_scaling,torch.zeros_like(model.d_scaling))),"finite":bool(torch.isfinite(model.canonical_xyz).all()),"delta_mean_norm":float(delta_norm.mean()),"delta_p95_norm":float(delta_norm.quantile(.95)),"knn_edge_length_ratio_median":float(ratio.median()),"knn_edge_length_ratio_p05":float(ratio.quantile(.05)),"knn_edge_length_ratio_p95":float(ratio.quantile(.95))}
    if a.exported_ply:
        exported=xyz_from_ply(a.exported_ply); report["exported_reload_count"] = int(len(exported)); report["exported_reload_matches_canonical"] = bool(torch.allclose(exported,model.canonical_xyz,atol=1e-5))
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)),exist_ok=True); open(a.output_path,"w").write(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
