"""Source Gaussian visibility estimates for image observations."""
import numpy as np


def project_points(points, full_proj_transform, image_width, image_height):
    import torch
    points = torch.as_tensor(points)
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    clip = torch.cat([points, ones], dim=1) @ full_proj_transform
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None].clamp_min(1e-8)
    xy = torch.stack([(ndc[:, 0] * .5 + .5) * float(image_width),
                      (1 - (ndc[:, 1] * .5 + .5)) * float(image_height)], dim=1)
    in_frame = (w > 0) & (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1)
    return xy, in_frame, w


def camera_space_depth(points, world_view_transform):
    import torch
    points = torch.as_tensor(points)
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    camera = torch.cat([points, ones], dim=1) @ world_view_transform
    return camera[:, 2]


def estimate_projected_visibility(xy, depth, source_mask, foreground_mask=None,
                                  depth_tolerance=0.15, bin_size=4):
    """Approximate visibility with source mask and a coarse depth z-buffer.

    The renderer does not expose Gaussian IDs on the CPU path. This approximate
    estimate uses the source silhouette and keeps all layers within a soft
    relative depth tolerance of the nearest Gaussian per coarse pixel bin.
    """
    xy = np.asarray(xy, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    mask = np.asarray(source_mask).astype(bool)
    h, w = mask.shape[-2:]
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & np.isfinite(depth) & (depth > 0)
    valid &= mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    if foreground_mask is not None:
        valid &= np.asarray(foreground_mask).reshape(-1).astype(bool)
    bins = {}
    for i in np.flatnonzero(valid):
        key = (int(x[i] // max(1, bin_size)), int(y[i] // max(1, bin_size)))
        bins.setdefault(key, []).append(int(i))
    result = np.zeros(len(xy), dtype=bool)
    for indices in bins.values():
        z = min(depth[indices])
        for i in indices:
            result[i] = depth[i] <= z * (1.0 + depth_tolerance)
    return result
