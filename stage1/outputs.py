"""Stage 1 output serialization and metadata contracts."""
import os

import torch


def save_stage1_delta(path, gaussians, d_xyz, d_rotation, d_scaling, metadata, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "d_xyz": d_xyz.detach().cpu(),
        "d_scaling": d_scaling.detach().cpu(),
        "d_rotation": d_rotation.detach().cpu(),
        "source_xyz": gaussians.get_xyz.detach().cpu(),
        "metadata": metadata or {},
    }
    try:
        payload["source_scaling"] = gaussians.get_scaling.detach().cpu()
    except Exception:
        pass
    if extra:
        payload.update(extra)
    torch.save(payload, path)
