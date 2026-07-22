"""Graph and magnitude regularizers separated from the Stage 1 loop."""
import warnings

import torch


def deformation_magnitude_loss(d_xyz, d_scaling=None, mask=None):
    values = d_xyz.square().sum(dim=-1)
    if d_scaling is not None:
        values = values + d_scaling.square().sum(dim=-1)
    if mask is not None and mask.any():
        return values[mask].mean()
    return values.mean()


def graph_smoothness_loss(xyz, d_xyz, k=16, sample=4096, chunk=512):
    try:
        n = xyz.shape[0]
        if n <= 1:
            return torch.zeros((), device=xyz.device)
        count = min(sample, n)
        sample_idx = torch.randperm(n, device=xyz.device)[:count]
        values = []
        for start in range(0, count, chunk):
            idx = sample_idx[start:start + chunk]
            distances = torch.cdist(xyz[idx].detach(), xyz.detach())
            neighbors = torch.topk(distances, k=min(k + 1, n), largest=False).indices[:, 1:]
            values.append((d_xyz[idx, None, :] - d_xyz[neighbors]).norm(dim=-1).mean())
        return torch.stack(values).mean()
    except Exception as exc:
        warnings.warn("Skipping graph smoothness because KNN failed: {}".format(exc))
        return torch.zeros((), device=xyz.device)
