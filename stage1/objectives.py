"""Image and observation objectives used by Stage 1."""
import torch
import torch.nn.functional as F

from correspondence.losses import confidence_weighted_3d_huber, projected_motion_huber


def masked_l1_loss(image, target, mask):
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask[None], size=image.shape[-2:], mode="nearest")[0]
    denom = mask.sum().clamp_min(1.0) * image.shape[0]
    return ((image - target).abs() * mask).sum() / denom


__all__ = ["confidence_weighted_3d_huber", "projected_motion_huber", "masked_l1_loss"]
