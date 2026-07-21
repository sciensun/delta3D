"""Visibility and match-format validation helpers."""
import torch


def validate_view_matches(matches, num_points=None):
    required = {"source_xy", "target_xy", "source_gaussian_id", "feature_confidence", "visibility"}
    missing = required.difference(matches)
    if missing: raise ValueError("view match payload missing {}".format(sorted(missing)))
    source = torch.as_tensor(matches["source_xy"]); target = torch.as_tensor(matches["target_xy"])
    ids = torch.as_tensor(matches["source_gaussian_id"]).long().flatten()
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or ids.shape[0] != source.shape[0]:
        raise ValueError("source_xy, target_xy, and source_gaussian_id shape mismatch")
    if num_points is not None and ((ids < 0) | (ids >= num_points)).any():
        raise ValueError("source Gaussian ID out of range")
    return True


def foreground_visibility(visibility, source_mask=None, target_mask=None):
    visible = torch.as_tensor(visibility).bool()
    if source_mask is not None: visible = visible & torch.as_tensor(source_mask).bool()
    if target_mask is not None: visible = visible & torch.as_tensor(target_mask).bool()
    return visible
