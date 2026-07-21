#!/usr/bin/env python3
"""Prepare deterministic weak-target split roots without inventing camera matches."""
import argparse
import json
import os
import shutil
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("-s","--source_path",required=True); p.add_argument("--target_image_root",required=True)
    p.add_argument("--out_root",required=True); p.add_argument("--include_missing",action="store_true")
    a=p.parse_args(); os.makedirs(a.out_root,exist_ok=True)
    source=json.load(open(os.path.join(a.source_path,"transforms_train.json")))
    angles={}
    for f in source.get("frames",[]):
        name=os.path.basename(f.get("file_path",""));
        if "elev000_az" in name:
            az=int(name.split("_az")[-1].split(".")[0]); angles[az]=name
    records=[]
    for idx in range(8):
        az=idx*45; target="{:02d}_standard.png".format(idx+1)
        records.append({"index":idx,"azimuth":az,"target":target,"source_image_name":angles.get(az),"matched":az in angles})
    for label, indices in {"subset_A":[0,2,4,6],"subset_B":[1,3,5,7]}.items():
        root=os.path.join(a.out_root,label); os.makedirs(root,exist_ok=True)
        selected=[records[i] for i in indices]
        for r in selected:
            src=os.path.join(a.target_image_root,r["target"])
            if not os.path.isfile(src) and r.get("source_image_name"):
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = os.path.join(a.target_image_root, r["source_image_name"] + ext)
                    if os.path.isfile(candidate):
                        src = candidate
                        break
            if os.path.isfile(src): shutil.copy2(src,os.path.join(root,r["target"]))
        with open(os.path.join(root,"split_manifest.json"),"w") as f: json.dump({"split":label,"records":selected},f,indent=2)
        missing=[r for r in selected if not r["matched"]]
        print(label,"targets:",len(selected),"matched source cameras:",len(selected)-len(missing),"missing:",missing)
    with open(os.path.join(a.out_root,"all_key8_match_manifest.json"),"w") as f: json.dump(records,f,indent=2)
    missing=[r for r in records if not r["matched"]]
    if missing and not a.include_missing:
        print("WARNING: source dataset has no exact camera for:",missing)
        print("No target was remapped; regenerate the missing source view or exclude it explicitly.")

if __name__=="__main__": main()
