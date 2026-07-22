"""Metrics for evaluating image-derived observations against hidden GT."""
import numpy as np
import torch


def endpoint_metrics(observed_xy, oracle_xy, valid):
    observed_xy = torch.as_tensor(observed_xy).float()
    oracle_xy = torch.as_tensor(oracle_xy).float()
    valid = torch.as_tensor(valid).bool()
    error = torch.linalg.vector_norm(observed_xy - oracle_xy, dim=-1)
    values = error[valid]
    if values.numel() == 0:
        return {"count": 0, "median_epe": None, "mean_epe": None,
                "p90_epe": None, "pck@1": 0.0, "pck@3": 0.0,
                "pck@5": 0.0, "pck@10": 0.0}
    return {
        "count": int(values.numel()),
        "median_epe": float(values.median()),
        "mean_epe": float(values.mean()),
        "p90_epe": float(torch.quantile(values, 0.9)),
        "pck@1": float((values <= 1).float().mean()),
        "pck@3": float((values <= 3).float().mean()),
        "pck@5": float((values <= 5).float().mean()),
        "pck@10": float((values <= 10).float().mean()),
    }


def confidence_calibration(confidence, error, valid, bins=5):
    confidence = torch.as_tensor(confidence).float()[torch.as_tensor(valid).bool()]
    error = torch.as_tensor(error).float()[torch.as_tensor(valid).bool()]
    result = []
    if confidence.numel() == 0:
        return result
    for lo, hi in zip(torch.linspace(0, 1, bins + 1)[:-1], torch.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if mask.any():
            result.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
                           "mean_confidence": float(confidence[mask].mean()),
                           "mean_error": float(error[mask].mean())})
    return result


def _basic(values):
    values = values.flatten()
    if values.numel() == 0:
        return {"count": 0, "median": None, "mean": None, "p90": None,
                "pck@1": 0.0, "pck@3": 0.0, "pck@5": 0.0, "pck@10": 0.0}
    return {"count": int(values.numel()), "median": float(values.median()),
            "mean": float(values.mean()), "p90": float(torch.quantile(values, .9)),
            "pck@1": float((values <= 1).float().mean()),
            "pck@3": float((values <= 3).float().mean()),
            "pck@5": float((values <= 5).float().mean()),
            "pck@10": float((values <= 10).float().mean())}


def confidence_precision_coverage(error, confidence, candidate, percentages=(.1, .25, .5, .75, 1.0)):
    error = torch.as_tensor(error).float()
    confidence = torch.as_tensor(confidence).float()
    candidate = torch.as_tensor(candidate).bool()
    e, c = error[candidate], confidence[candidate]
    if e.numel() == 0:
        return {str(int(p * 100)): {"count": 0, "coverage": 0.0} for p in percentages}
    order = torch.argsort(c, descending=True)
    output = {}
    for p in percentages:
        count = max(1, int(round(float(p) * e.numel())))
        chosen = e[order[:count]]
        output[str(int(p * 100))] = {"count": int(count), "coverage": float(count / e.numel()),
                                     **_basic(chosen)}
    return output


def stratified_displacement_metrics(error, displacement, valid, bins=None):
    bins = bins or [(0, 1), (1, 3), (3, 5), (5, 10), (10, float("inf"))]
    error, displacement = torch.as_tensor(error).float(), torch.as_tensor(displacement).float()
    valid = torch.as_tensor(valid).bool()
    output = {}
    for lo, hi in bins:
        selected = valid & (displacement >= lo) & (displacement < hi)
        output["{}-{}px".format(lo, "inf" if hi == float("inf") else hi)] = _basic(error[selected])
    return output


def evaluate_view_observations(observed_xy, oracle_xy, accepted, candidates,
                               confidence, source_xy, foreground, active):
    observed_xy = torch.as_tensor(observed_xy).float()
    oracle_xy = torch.as_tensor(oracle_xy).float()
    accepted = torch.as_tensor(accepted).bool()
    candidates = torch.as_tensor(candidates).bool()
    confidence = torch.as_tensor(confidence).float()
    source_xy = torch.as_tensor(source_xy).float()
    foreground, active = torch.as_tensor(foreground).bool(), torch.as_tensor(active).bool()
    error = torch.linalg.vector_norm(observed_xy - oracle_xy, dim=-1)
    displacement = torch.linalg.vector_norm(oracle_xy - source_xy, dim=-1)
    candidate_fg = candidates & foreground
    accepted_fg = accepted & foreground
    candidate_active = candidates & active
    accepted_active = accepted & active
    candidate_inactive = candidates & foreground & ~active
    accepted_inactive = accepted & foreground & ~active
    result = {
        "all_gaussian_support_coverage": float(candidates.float().mean()),
        "foreground_support_coverage": float(candidate_fg.float().sum() / foreground.float().sum().clamp_min(1)),
        "active_region_coverage": float(candidate_active.float().sum() / active.float().sum().clamp_min(1)),
        "inactive_region_coverage": float(candidate_inactive.float().sum() / (foreground & ~active).float().sum().clamp_min(1)),
        "visible_candidate_count": int(candidates.sum()),
        "accepted_match_count": int(accepted.sum()),
        "accepted_match_recall": float(accepted[candidates].float().mean()) if candidates.any() else 0.0,
        "active_accepted_match_recall": float(accepted[candidate_active].float().mean()) if candidate_active.any() else 0.0,
        "inactive_false_motion_magnitude": float(displacement[accepted_inactive].mean()) if accepted_inactive.any() else 0.0,
        "accepted": _basic(error[accepted]),
        "foreground": _basic(error[accepted_fg]),
        "active": _basic(error[accepted_active]),
        "inactive": _basic(error[accepted_inactive]),
        "active_stratified_displacement": stratified_displacement_metrics(error, displacement, accepted_active),
        "confidence_precision_coverage": confidence_precision_coverage(error, confidence, candidates),
        "zero_motion": _basic(displacement[candidates]),
        "zero_motion_active": _basic(displacement[candidate_active]),
    }
    return result
