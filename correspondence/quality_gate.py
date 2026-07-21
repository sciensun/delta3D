"""Quality metrics for deciding whether correspondence can supervise Stage 1."""
import torch


def correspondence_quality(bundle):
    bundle.validate()
    valid = bundle.valid_mask
    displacement = bundle.target_xyz - bundle.source_xyz
    return {
        "num_gaussians": int(valid.numel()), "valid_gaussians": int(valid.sum()),
        "coverage": float(valid.float().mean()),
        "mean_confidence_valid": float(bundle.confidence[valid].mean()) if valid.any() else 0.0,
        "mean_residual_3d_valid": float(bundle.residual_3d[valid].mean()) if valid.any() else 0.0,
        "mean_directional_variance_valid": float(bundle.directional_variance[valid].mean()) if valid.any() else 0.0,
        "mean_displacement_norm_valid": float(displacement[valid].norm(dim=-1).mean()) if valid.any() else 0.0,
        "invalid_or_unmatched_gaussians": int((~valid).sum()),
    }
