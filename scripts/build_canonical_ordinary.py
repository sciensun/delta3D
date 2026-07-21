#!/usr/bin/env python3
"""Build G_ord_canon on the unchanged G_sty Gaussian bank."""
import argparse, json, os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.canonical_ordinary_model import CanonicalOrdinaryModel

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source_xyz_path",required=True); p.add_argument("--delta_path",required=True); p.add_argument("--output_dir",required=True); p.add_argument("--correspondence_bundle",default=None); p.add_argument("--source_model",default=None); p.add_argument("--reference_model",default=None); a=p.parse_args()
    source=torch.load(a.source_xyz_path,map_location="cpu"); source=source["source_xyz"] if isinstance(source,dict) else source
    delta=torch.load(a.delta_path,map_location="cpu")
    bundle=None
    if a.correspondence_bundle:
        from correspondence.schema import CorrespondenceBundle
        bundle=CorrespondenceBundle.load(a.correspondence_bundle)
    model=CanonicalOrdinaryModel.from_delta(source,delta,bundle)
    os.makedirs(a.output_dir,exist_ok=True); model.save(os.path.join(a.output_dir,"canonical_ordinary.pt")); model.export_delta(os.path.join(a.output_dir,"Delta_star.pt"))
    meta={"source_model":a.source_model,"reference_model":a.reference_model,"num_gaussians":model.num_gaussians,"fixed_bank":True,"gaussian_order_unchanged":True,"d_scaling_disabled":True,"background_delta_zero":True}
    with open(os.path.join(a.output_dir,"canonical_metadata.json"),"w") as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
