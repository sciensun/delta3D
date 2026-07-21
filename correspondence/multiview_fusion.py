"""Confidence-weighted robust fusion of target xyz candidates."""
import torch


def fuse_target_candidates(candidate_target_xyz, valid, confidence=None):
    candidates = torch.as_tensor(candidate_target_xyz).float()
    valid = torch.as_tensor(valid).bool()
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or valid.shape != candidates.shape[:2]:
        raise ValueError("candidate_target_xyz must be [V,N,3] and valid [V,N]")
    weights = torch.ones_like(valid, dtype=candidates.dtype) if confidence is None else torch.as_tensor(confidence).float()
    weights = weights * valid.float()
    denom = weights.sum(0).clamp_min(1e-8)
    target = (candidates * weights[..., None]).sum(0) / denom[:, None]
    support = valid.sum(0)
    residual = ((candidates - target[None]).square().sum(-1) * valid.float()).sum(0) / support.clamp_min(1)
    directions = candidates / candidates.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mean_dir = (directions * weights[..., None]).sum(0) / denom[:, None]
    variance = ((1 - (directions * mean_dir[None]).sum(-1)) * weights).sum(0) / denom
    confidence_out = weights.sum(0) / max(1, candidates.shape[0])
    valid_out = support > 0
    return {"target_xyz": target, "valid_mask": valid_out, "confidence": confidence_out,
            "support_count": support, "residual_3d": residual, "directional_variance": variance}
