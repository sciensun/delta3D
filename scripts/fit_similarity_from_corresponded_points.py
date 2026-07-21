#!/usr/bin/env python3
"""Fit similarity from already paired sparse points or anchor JSON."""
import argparse
import json
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correspondence.alignment import fit_similarity_from_corresponded_points, apply_similarity, transform_to_json


def load_points(path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        return x.get("source_xyz", x.get("xyz")), x.get("target_xyz")
    raise ValueError("point payload must contain source_xyz and target_xyz")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source_points", required=True); p.add_argument("--target_points", required=True)
    p.add_argument("--output_transform", required=True); p.add_argument("--output_aligned", default=None); a=p.parse_args()
    source=load_points(a.source_points)[0]; target=load_points(a.target_points)[1] if load_points(a.target_points)[1] is not None else load_points(a.target_points)[0]
    transform=fit_similarity_from_corresponded_points(source,target)
    payload=transform_to_json(transform); payload["before_rmse"]=float((source-target).square().mean().sqrt()); aligned=apply_similarity(source,transform); payload["after_rmse"]=float((aligned-target).square().mean().sqrt())
    os.makedirs(os.path.dirname(os.path.abspath(a.output_transform)),exist_ok=True)
    with open(a.output_transform,"w") as f: json.dump(payload,f,indent=2)
    if a.output_aligned: torch.save({"xyz":aligned,"target_xyz":target,"metadata":payload},a.output_aligned)
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
