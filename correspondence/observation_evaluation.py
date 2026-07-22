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
