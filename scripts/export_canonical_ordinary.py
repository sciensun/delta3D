#!/usr/bin/env python3
"""Export fixed-bank canonical ordinary PLY without changing Gaussian schema."""
import argparse, os, sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.canonical_ordinary_model import CanonicalOrdinaryModel

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source_ply",required=True); p.add_argument("--canonical_path",required=True); p.add_argument("--output_ply",required=True); a=p.parse_args()
    x=torch.load(a.canonical_path,map_location="cpu"); model=CanonicalOrdinaryModel(x["source_xyz"],x["d_xyz"],x.get("d_scaling"),x.get("foreground_mask"),x.get("correspondence_confidence"),x.get("metadata")); model.export_ply(a.source_ply,a.output_ply); print("exported:",a.output_ply)
if __name__=="__main__": main()
