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
        # Normalize by valid confidence. Otherwise a low-confidence template
        # can appear artificially close simply because it contributes less.
        score = (residual * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
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


def _region_metrics(pred, target, mask):
    mask = torch.as_tensor(mask).bool()
    if not bool(mask.any()):
        return {"count": 0, "cosine": 0.0, "energy_ratio": 0.0,
                "explained_variance": 0.0, "magnitude_pearson": 0.0,
                "magnitude_spearman": 0.0}
    p, t = pred[mask], target[mask]
    diff = p - t
    denom = t.square().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(p.flatten()[None], t.flatten()[None]).item()
    norms_p, norms_t = p.norm(dim=-1), t.norm(dim=-1)
    centered = (t - t.mean(0)).square().sum().clamp_min(1e-12)
    return {
        "count": int(mask.sum()),
        "cosine": float(cosine),
        "energy_ratio": float(p.square().sum() / denom),
        "explained_variance": float(1.0 - diff.square().sum() / centered),
        "magnitude_pearson": _safe_corr(norms_p, norms_t),
        "magnitude_spearman": _safe_corr(torch.argsort(torch.argsort(norms_p)), torch.argsort(torch.argsort(norms_t))),
        "pred_norm_mean": float(norms_p.mean()),
        "target_norm_mean": float(norms_t.mean()),
    }


def delta_metrics(pred, target, active_mask=None, source_xyz=None,
                  foreground_mask=None, style_region_mask=None):
    """Return distinct full-bank, foreground, and active-region metrics."""
    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()
    n = pred.shape[0]
    full = torch.ones(n, dtype=torch.bool)
    fg = full if foreground_mask is None else torch.as_tensor(foreground_mask).bool()
    active = fg if active_mask is None else torch.as_tensor(active_mask).bool()
    result = {
        "global": _region_metrics(pred, target, full),
        "foreground": _region_metrics(pred, target, fg),
        "active": _region_metrics(pred, target, active),
        "background_energy": float(pred[~fg].square().sum()),
        "active_energy_percent_of_foreground": float(pred[active].square().sum() / pred[fg].square().sum().clamp_min(1e-12) * 100),
        "style_region_leakage": float(pred[fg & ~active].square().sum() / pred[fg].square().sum().clamp_min(1e-12)),
    }
    # Backward-compatible aliases point to active values but are explicitly
    # labelled as legacy; callers should use result[region][metric].
    result["global_cosine"] = result["global"]["cosine"]
    result["active_cosine"] = result["active"]["cosine"]
    result["energy_ratio"] = result["active"]["energy_ratio"]
    result["explained_variance"] = result["active"]["explained_variance"]
    result["active_count"] = result["active"]["count"]
    return result


def factorize_candidates(deltas, confidence=None, nuisance_features=None, trim_fraction=0.2,
                         geometric_iterations=32, robust_iterations=8):
    deltas = torch.as_tensor(deltas).float()
    result = {
        "single_template": deltas[0],
        "mean": weighted_mean(deltas, confidence),
        "median": coordinate_median(deltas, confidence),
        "geometric_median": geometric_median(deltas, confidence, iterations=geometric_iterations),
        "robust_shared": robust_shared(deltas, confidence, trim_fraction=trim_fraction, iterations=robust_iterations),
    }
    if nuisance_features is not None:
        shared, nuisance, reconstructed = nuisance_regression(deltas, nuisance_features, confidence)
        result["nuisance_regression"] = shared
        result["nuisance_components"] = nuisance
        result["reconstructed"] = reconstructed
    return result


def structured_no_label_factorization(deltas, confidence=None, rank=2,
                                      iterations=20, lambda_sparse=0.0,
                                      neighbors=None, graph_blend=0.05,
                                      foreground_mask=None):
    """Alternating low-rank shared/nuisance factorization without labels.

    The intercept is S, mode fields are U, coefficients A are centered over
    templates, and E is the residual. This is intentionally deterministic and
    CPU-safe; it is not a neural model and never reads a hidden teacher.
    """
    d = torch.as_tensor(deltas).float()
    r, n, c = d.shape
    if rank >= r:
        rank = max(1, r - 1)
    w = _as_confidence(d, confidence)
    foreground_mask = torch.ones(n, dtype=torch.bool) if foreground_mask is None else torch.as_tensor(foreground_mask).bool()
    d[:, ~foreground_mask] = 0
    s = weighted_mean(d, w)
    centered = d - s[None]
    # Deterministic initialization from the low-dimensional template axis.
    cov = torch.einsum("rnc,snc->rs", centered, centered)
    _, vec = torch.linalg.eigh(cov)
    a = vec[:, -rank:].float()
    a = a - a.mean(0, keepdim=True)
    u = torch.zeros((rank, n, c))
    eye = 1e-5 * torch.eye(rank)
    for _ in range(iterations):
        aa = a.T @ a + eye
        u = torch.einsum("kr,rnc->knc", torch.linalg.solve(aa, a.T), d - s[None])
        if neighbors is not None and graph_blend > 0:
            u = (1 - graph_blend) * u + graph_blend * u[:, neighbors].mean(2)
        uu = torch.einsum("knc,lnc->kl", u, u) + eye
        a = torch.einsum("rnc,knc,kl->rl", d - s[None], u, torch.linalg.inv(uu))
        a = a - a.mean(0, keepdim=True)
        s = weighted_mean(d - torch.einsum("rk,knc->rnc", a, u), w)
        if neighbors is not None and graph_blend > 0:
            s = (1 - graph_blend) * s + graph_blend * s[neighbors].mean(1)
        s[~foreground_mask] = 0
        u[:, ~foreground_mask] = 0
    reconstructed = s[None] + torch.einsum("rk,knc->rnc", a, u)
    residual = d - reconstructed
    reconstructed[:, ~foreground_mask] = 0
    residual[:, ~foreground_mask] = 0
    if lambda_sparse > 0:
        residual = torch.sign(residual) * torch.relu(residual.abs() - lambda_sparse)
    return {"shared": s, "nuisance_modes": u, "coefficients": a,
            "residual": residual, "reconstructed": reconstructed,
            "rank": int(rank), "convergence_residual_norm": float(residual.norm())}
