"""CPU-friendly shared-style/template-nuisance factorization.

This module operates on already mined candidate deltas. It does not inspect
images or target geometry and therefore cannot manufacture correspondence.
"""
from typing import Dict, Optional

import torch


def _as_confidence(deltas, confidence=None):
    r, n, _ = deltas.shape
    if confidence is None:
        return torch.ones((r, n), dtype=deltas.dtype, device=deltas.device)
    return torch.as_tensor(confidence, dtype=deltas.dtype, device=deltas.device).clamp_min(0)


def weighted_mean(deltas, confidence=None):
    w = _as_confidence(deltas, confidence)
    return (deltas * w[..., None]).sum(0) / w.sum(0).clamp_min(1e-8)[..., None]


def coordinate_median(deltas, confidence=None):
    # A weighted median is more robust, but the unweighted coordinate median is
    # deterministic and useful as a baseline for balanced template variants.
    deltas = torch.as_tensor(deltas)
    return deltas.median(dim=0).values


def geometric_median(deltas, confidence=None, iterations=32, eps=1e-6):
    """Per-Gaussian weighted geometric median via Weiszfeld iterations."""
    deltas = torch.as_tensor(deltas).float()
    w = _as_confidence(deltas, confidence)
    estimate = weighted_mean(deltas, w)
    for _ in range(iterations):
        distance = (deltas - estimate[None]).norm(dim=-1).clamp_min(eps)
        ww = w / distance
        estimate = (deltas * ww[..., None]).sum(0) / ww.sum(0).clamp_min(eps)[..., None]
    return estimate


def robust_shared(deltas, confidence=None, trim_fraction=0.2, iterations=8):
    """Trim high-residual templates, then recompute a weighted shared delta."""
    deltas = torch.as_tensor(deltas).float()
    w = _as_confidence(deltas, confidence)
    estimate = geometric_median(deltas, w, iterations=16)
    keep_count = max(1, int(round(deltas.shape[0] * (1.0 - trim_fraction))))
    for _ in range(iterations):
        residual = (deltas - estimate[None]).norm(dim=-1)
        score = (residual * w).mean(dim=1)
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep[torch.argsort(score)[:keep_count]] = True
        masked = w * keep[:, None].float()
        estimate = geometric_median(deltas, masked, iterations=12)
    return estimate


def nuisance_regression(deltas, nuisance_features, confidence=None, ridge=1e-5):
    """Fit Delta_r = S + U a_r with zero-centered nuisance features."""
    deltas = torch.as_tensor(deltas).float()
    a = torch.as_tensor(nuisance_features, dtype=deltas.dtype, device=deltas.device)
    if a.ndim != 2 or a.shape[0] != deltas.shape[0]:
        raise ValueError("nuisance_features must be [R,K]")
    a = a - a.mean(0, keepdim=True)
    x = torch.cat([torch.ones((a.shape[0], 1), dtype=a.dtype, device=a.device), a], dim=1)
    w = _as_confidence(deltas, confidence).transpose(0, 1)  # [N,R]
    # Batched weighted least squares over Gaussian and xyz dimensions.
    y = deltas.permute(1, 2, 0)  # [N,3,R]
    normal = torch.einsum("rk,nr,rl->nkl", x, w, x)
    normal = normal + ridge * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)[None]
    rhs = torch.einsum("rk,nr,ncr->nkc", x, w, y)
    beta = torch.linalg.solve(normal, rhs)  # [N,K+1,3]
    shared = beta[:, 0]
    nuisance = beta[:, 1:].permute(1, 0, 2)  # [K,N,3]
    reconstructed = torch.einsum("rk,knc->rnc", a, nuisance) + shared[None]
    return shared, nuisance, reconstructed


def _safe_corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2 or float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def delta_metrics(pred, target, active_mask=None, source_xyz=None):
    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()
    mask = torch.ones(pred.shape[0], dtype=torch.bool) if active_mask is None else torch.as_tensor(active_mask).bool()
    p, t = pred[mask], target[mask]
    diff = p - t
    cosine = torch.nn.functional.cosine_similarity(p.flatten()[None], t.flatten()[None]).item() if p.numel() else 0.0
    denom = t.square().sum().clamp_min(1e-12)
    energy = float(p.square().sum() / denom)
    explained = 1.0 - float(diff.square().sum() / denom)
    norms_p, norms_t = p.norm(dim=-1), t.norm(dim=-1)
    return {
        "global_cosine": float(cosine),
        "active_cosine": float(cosine),
        "energy_ratio": energy,
        "explained_variance": explained,
        "magnitude_pearson": _safe_corr(norms_p, norms_t),
        "magnitude_spearman": _safe_corr(torch.argsort(torch.argsort(norms_p)), torch.argsort(torch.argsort(norms_t))),
        "active_count": int(mask.sum()),
        "pred_norm_mean": float(norms_p.mean()) if norms_p.numel() else 0.0,
        "target_norm_mean": float(norms_t.mean()) if norms_t.numel() else 0.0,
        "foreground_energy_percent": float(pred[mask].square().sum() / pred.square().sum().clamp_min(1e-12) * 100),
    }


def factorize_candidates(deltas, confidence=None, nuisance_features=None, trim_fraction=0.2):
    deltas = torch.as_tensor(deltas).float()
    result = {
        "single_template": deltas[0],
        "mean": weighted_mean(deltas, confidence),
        "median": coordinate_median(deltas, confidence),
        "geometric_median": geometric_median(deltas, confidence),
        "robust_shared": robust_shared(deltas, confidence, trim_fraction=trim_fraction),
    }
    if nuisance_features is not None:
        shared, nuisance, reconstructed = nuisance_regression(deltas, nuisance_features, confidence)
        result["nuisance_regression"] = shared
        result["nuisance_components"] = nuisance
        result["reconstructed"] = reconstructed
    return result
