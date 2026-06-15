import math
import warnings

import torch


def fourier_encode_xyz(xyz, num_pos_freqs=6):
    freqs = (2.0 ** torch.arange(num_pos_freqs, device=xyz.device, dtype=xyz.dtype)).view(1, -1, 1)
    xb = xyz[:, None, :] * freqs
    return torch.cat([xyz, torch.sin(xb).flatten(1), torch.cos(xb).flatten(1)], dim=-1)


def build_knn(xyz, k=32, query_xyz=None, chunk_size=1024):
    """Return indices of k nearest neighbors for each query point."""
    xyz_detached = xyz.detach()
    query = xyz_detached if query_xyz is None else query_xyz.detach()
    k_eff = min(k + (1 if query_xyz is None else 0), xyz_detached.shape[0])
    chunks = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start : start + chunk_size]
        dist = torch.cdist(q, xyz_detached)
        idx = torch.topk(dist, k=k_eff, largest=False).indices
        if query_xyz is None and idx.shape[1] > 1:
            idx = idx[:, 1:]
        chunks.append(idx[:, :k])
    return torch.cat(chunks, dim=0)


def _safe_normalize(x, eps=1e-8):
    return x / (x.abs().mean(dim=0, keepdim=True) + eps)


def compute_local_geometry_features(xyz, scaling=None, opacity=None, k=32, num_pos_freqs=6, chunk_size=1024):
    """Compute per-Gaussian local features for B(F_i, delta_z) distillation."""
    device = xyz.device
    dtype = xyz.dtype
    feats = [fourier_encode_xyz(xyz, num_pos_freqs=num_pos_freqs)]

    if scaling is not None:
        feats.append(torch.log(torch.clamp(scaling.detach(), min=1e-6)))
    if opacity is not None:
        feats.append(opacity.detach())

    try:
        knn = build_knn(xyz, k=k, chunk_size=chunk_size)
        eigvals_all = []
        density_all = []
        for start in range(0, xyz.shape[0], chunk_size):
            idx = knn[start : start + chunk_size]
            neigh = xyz[idx]
            center = xyz[start : start + idx.shape[0], None, :]
            centered = neigh - center
            cov = centered.transpose(1, 2).matmul(centered) / max(idx.shape[1] - 1, 1)
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0).flip(dims=[1])
            rk = centered.norm(dim=-1).max(dim=1).values.clamp_min(1e-6)
            density = torch.full_like(rk, float(idx.shape[1])) / (rk ** 3)
            eigvals_all.append(eigvals)
            density_all.append(density[:, None])
        eigvals = torch.cat(eigvals_all, dim=0)
        density = torch.cat(density_all, dim=0)
    except Exception as exc:
        warnings.warn("KNN/local covariance failed; using zero local geometry features: {}".format(exc))
        eigvals = torch.zeros((xyz.shape[0], 3), device=device, dtype=dtype)
        density = torch.zeros((xyz.shape[0], 1), device=device, dtype=dtype)

    l1 = eigvals[:, 0:1].clamp_min(1e-8)
    l2 = eigvals[:, 1:2]
    l3 = eigvals[:, 2:3]
    eig_sum = eigvals.sum(dim=1, keepdim=True).clamp_min(1e-8)
    local = torch.cat(
        [
            _safe_normalize(eigvals),
            (l1 - l2) / l1,
            (l2 - l3) / l1,
            l3 / l1,
            l3 / eig_sum,
            torch.log1p(density).clamp(max=20.0) / 20.0,
        ],
        dim=-1,
    )
    feats.append(local)
    return torch.cat(feats, dim=-1)
