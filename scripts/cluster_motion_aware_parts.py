#!/usr/bin/env python3
"""Cluster foreground Gaussians using geometry plus consensus motion."""
import argparse, json, os
import numpy as np
import torch


def write_color_ply(path, xyz, labels):
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(7)
    palette = rng.integers(32, 240, size=(max(int(labels.max()) + 1, 1), 3), dtype=np.uint8)
    colors = np.zeros((len(xyz), 3), dtype=np.uint8)
    valid = labels >= 0
    colors[valid] = torch.from_numpy(palette[labels[valid].numpy()]).numpy()
    data = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz.numpy().T
    data["red"], data["green"], data["blue"] = colors.T
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source_xyz_path",required=True); p.add_argument("--foreground_mask_path",required=True)
    p.add_argument("--delta_path",required=True); p.add_argument("--output_dir",required=True); p.add_argument("--num_parts",type=int,default=16)
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    xyz_payload=torch.load(a.source_xyz_path,map_location="cpu")
    xyz=(xyz_payload["source_xyz"] if isinstance(xyz_payload,dict) else xyz_payload).float()
    fg=torch.load(a.foreground_mask_path,map_location="cpu").bool().flatten(); d=torch.load(a.delta_path,map_location="cpu")["d_xyz"].float()
    ids=torch.where(fg)[0]; x=xyz[ids]; dn=d[ids].norm(dim=-1,keepdim=True); du=d[ids]/dn.clamp_min(1e-8); x=(x-x.mean(0))/(x.std(0)+1e-6); dn=torch.log1p(dn); feats=torch.cat([x,du,dn],1).numpy()
    try:
        from sklearn.cluster import MiniBatchKMeans
        labels_np=MiniBatchKMeans(n_clusters=a.num_parts,random_state=0,n_init=3,batch_size=4096).fit_predict(feats)
    except Exception as exc:
        print("WARNING: sklearn unavailable; using deterministic spatial bins:",exc)
        order=np.argsort(x[:,0].numpy()); labels_np=np.zeros(len(ids),dtype=np.int64); labels_np[order]=np.arange(len(ids))*a.num_parts//len(ids)
    labels=torch.full((len(xyz),),-1,dtype=torch.long); labels[ids]=torch.from_numpy(labels_np.astype(np.int64))
    torch.save(labels,os.path.join(a.output_dir,"motion_part_labels.pt"))
    write_color_ply(os.path.join(a.output_dir, "motion_part_clusters_color.ply"), xyz, labels)
    stats=[]
    for k in range(a.num_parts):
        m=labels==k
        if m.any(): stats.append({"part":k,"count":int(m.sum()),"mean_delta_norm":float(d[m].norm(dim=-1).mean()),"mean_delta":d[m].mean(0).tolist()})
    with open(os.path.join(a.output_dir,"motion_part_report.json"),"w") as f: json.dump({"num_parts":a.num_parts,"stats":stats},f,indent=2)
    print(json.dumps(stats,indent=2)); print("saved:",os.path.join(a.output_dir,"motion_part_labels.pt"))

if __name__=="__main__": main()
