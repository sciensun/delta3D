"""Rigid + preferred-uniform-scale alignment utilities."""
import torch


def fit_similarity_from_corresponded_points(source_xyz, target_xyz):
    source_xyz = torch.as_tensor(source_xyz).float(); target_xyz = torch.as_tensor(target_xyz).float()
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("paired points must both have shape [M,3]")
    source_center = source_xyz.mean(0); target_center = target_xyz.mean(0)
    xs = source_xyz - source_center; yt = target_xyz - target_center
    u, singular, vh = torch.linalg.svd((yt.T @ xs) / max(1, source_xyz.shape[0]))
    correction = torch.eye(3, dtype=source_xyz.dtype)
    if torch.det(u @ vh) < 0: correction[-1, -1] = -1
    rotation = u @ correction @ vh
    # The covariance above is normalized by the number of anchors, so the
    # source variance must use the same normalization.
    source_variance = xs.square().sum() / max(1, source_xyz.shape[0])
    scale = (singular * torch.diag(correction)).sum() / source_variance.clamp_min(1e-8)
    scale = scale.clamp_min(1e-8)
    translation = target_center - scale * (rotation @ source_center)
    return {"rotation": rotation, "scale": scale, "translation": translation}


def apply_similarity(xyz, transform):
    xyz = torch.as_tensor(xyz).float()
    return transform["scale"] * (xyz @ transform["rotation"].T) + transform["translation"]


def transform_to_json(transform):
    return {"rotation": transform["rotation"].tolist(), "scale": float(transform["scale"]),
            "translation": transform["translation"].tolist(), "method": "uniform_similarity"}
