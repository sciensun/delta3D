#!/usr/bin/env python3
"""Align an independent ordinary reference using sparse semantic anchor pairs."""
import argparse, json, os, sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.anchors import load_anchors, paired_anchor_points
from correspondence.alignment import fit_similarity_from_corresponded_points, apply_similarity, transform_to_json


def load_xyz(path):
    x=torch.load(path,map_location="cpu")
    if isinstance(x,dict):
        for k in ("xyz","source_xyz","target_xyz"):
            if k in x: return x[k].float()
    return x.float()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--reference_xyz_path",required=True); p.add_argument("--anchors_path",required=True); p.add_argument("--output_path",required=True); p.add_argument("--transform_path",required=True); a=p.parse_args()
    anchors=paired_anchor_points(load_anchors(a.anchors_path),min_confidence=0.0)
    source=torch.tensor([x["source_xyz"] for x in anchors],dtype=torch.float32); target=torch.tensor([x["target_xyz"] for x in anchors],dtype=torch.float32)
    transform=fit_similarity_from_corresponded_points(source,target); ref=load_xyz(a.reference_xyz_path); aligned=apply_similarity(ref,transform)
    os.makedirs(os.path.dirname(os.path.abspath(a.output_path)),exist_ok=True); torch.save({"xyz":aligned,"reference_xyz":ref,"alignment_transform":transform_to_json(transform),"metadata":{"nonrigid_warp":False}},a.output_path)
    with open(a.transform_path,"w") as f: json.dump(transform_to_json(transform),f,indent=2)
    print(json.dumps(transform_to_json(transform),indent=2))

if __name__=="__main__": main()
