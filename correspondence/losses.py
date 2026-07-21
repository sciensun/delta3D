"""Correspondence losses kept independent from the Stage 1 training script."""
import torch
import torch.nn.functional as F


def _masked_huber(pred, target, confidence, valid, delta):
    valid = torch.as_tensor(valid).bool().flatten()
    confidence = torch.as_tensor(confidence).float().flatten() * valid.float()
    denom = confidence.sum().clamp_min(1e-8)
    if not valid.any(): return pred.sum() * 0.0
    value = F.huber_loss(pred[valid], target[valid], reduction="none", delta=delta).mean(dim=-1)
    return (value * confidence[valid]).sum() / denom


def confidence_weighted_3d_huber(predicted_xyz, target_xyz, confidence, valid_mask, delta=0.01):
    return _masked_huber(predicted_xyz, target_xyz, confidence, valid_mask, delta)


def project_points(points, camera):
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    clip = torch.cat([points, ones], dim=1) @ camera.full_proj_transform
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)
    return torch.stack([(ndc[:, 0] * .5 + .5) * float(camera.image_width),
                        (1 - (ndc[:, 1] * .5 + .5)) * float(camera.image_height)], dim=1)


def projected_motion_huber(source_xyz, d_xyz, camera, target_xy, visibility, confidence, delta=1.0, oracle_target_xyz=None, valid_mask=None):
    source_xy = project_points(source_xyz, camera)
    if target_xy is None:
        if oracle_target_xyz is None: return d_xyz.sum() * 0.0
        target_xy = project_points(oracle_target_xyz, camera)
    predicted_xy = project_points(source_xyz + d_xyz, camera)
    target_motion = torch.as_tensor(target_xy, device=source_xyz.device, dtype=source_xyz.dtype) - source_xy
    predicted_motion = predicted_xy - source_xy
    valid = torch.as_tensor(visibility, device=source_xyz.device).bool().flatten()
    if valid_mask is not None: valid = valid & torch.as_tensor(valid_mask, device=source_xyz.device).bool().flatten()
    return _masked_huber(predicted_motion, target_motion, confidence, valid, delta)


def depth_consistency_loss(predicted_xyz, target_xyz, confidence, valid_mask, delta=0.01):
    return confidence_weighted_3d_huber(predicted_xyz, target_xyz, confidence, valid_mask, delta)
