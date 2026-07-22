"""Foreground, mutual-consistency, and confidence filters."""
import numpy as np


def sample_image(image, xy):
    image = np.asarray(image)
    h, w = image.shape[:2]
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return image[y, x]


def robust_patch_flow(flow, xy, radius=2, confidence=None, foreground=None,
                      radii=None):
    """Weighted robust flow over a local image footprint.

    `radii` is per-point when renderer projected radii are available; the
    fallback radius is an image neighborhood, not a Gaussian footprint.
    """
    flow = np.asarray(flow)
    offsets = [(dx, dy) for dx in range(-radius, radius + 1)
               for dy in range(-radius, radius + 1)]
    samples, weights = [], []
    h, w = flow.shape[:2]
    for dx, dy in offsets:
        p = xy + np.array([dx, dy], dtype=np.float32)
        x = np.clip(np.rint(p[:, 0]).astype(np.int64), 0, w - 1)
        y = np.clip(np.rint(p[:, 1]).astype(np.int64), 0, h - 1)
        sample = flow[y, x]
        weight = np.exp(-(dx * dx + dy * dy) / max(1.0, radius * radius))
        if confidence is not None:
            weight = weight * np.asarray(confidence)[y, x]
        if foreground is not None:
            weight = weight * np.asarray(foreground)[y, x].astype(np.float32)
        samples.append(sample)
        weights.append(np.full(len(xy), weight, dtype=np.float32))
    values = np.stack(samples, axis=0)
    weights = np.stack(weights, axis=0)
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(weights[..., None], order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    threshold = cumulative[-1:] * 0.5
    index = (cumulative >= threshold).argmax(axis=0)
    return np.take_along_axis(sorted_values, index[None, :, :], axis=0)[0]


def build_point_matches(xy, match_field, source_mask, target_mask, source_visible,
                        search_radius=2, max_cycle_error=3.0, min_confidence=0.15):
    xy = np.asarray(xy, dtype=np.float32)
    source_visible = np.asarray(source_visible).astype(bool)
    flow = robust_patch_flow(match_field.flow, xy, radius=search_radius,
                             confidence=getattr(match_field, "confidence", None),
                             foreground=source_mask)
    target_xy = xy + flow
    h, w = np.asarray(target_mask).shape[-2:]
    tx = np.rint(target_xy[:, 0]).astype(np.int64)
    ty = np.rint(target_xy[:, 1]).astype(np.int64)
    in_target = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    target_fg = np.zeros(len(xy), dtype=bool)
    target_fg[in_target] = np.asarray(target_mask)[ty[in_target], tx[in_target]].astype(bool)
    cycle = np.asarray(match_field.cycle_error)
    cycle_at_source = sample_image(cycle, xy).reshape(-1)
    confidence = np.exp(-cycle_at_source / 2.0).astype(np.float32)
    valid = source_visible & in_target & target_fg & (cycle_at_source <= max_cycle_error)
    valid &= confidence >= min_confidence
    return target_xy, valid, confidence, cycle_at_source
