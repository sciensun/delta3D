"""Track-level multi-view consistency, separate from per-view IRLS kernels."""
import torch


def leave_one_view_out_consensus(source_views, jacobians, target_xy, visibility,
                                 residual_threshold=3.0, min_inliers=2):
    """Reject views whose held-out reprojection disagrees with the other views.

    Exact two-view tracks are returned unchanged: there is no independent
    within-track consensus with only one view left for fitting.
    """
    sv = torch.as_tensor(source_views).float(); j = torch.as_tensor(jacobians).float()
    target = torch.as_tensor(target_xy).float(); vis = torch.as_tensor(visibility).bool()
    vcount, n = vis.shape; accepted = vis.clone(); errors = torch.full((vcount, n), float("nan"))
    for i in range(n):
        valid = torch.nonzero(vis[:, i]).flatten()
        if len(valid) < 3:
            continue
        for held in valid.tolist():
            fit_views = [x for x in valid.tolist() if x != held]
            normal = torch.zeros((3, 3)); rhs = torch.zeros(3)
            for v in fit_views:
                normal += j[v, i].T @ j[v, i]
                rhs += j[v, i].T @ (target[v, i] - sv[v, i])
            delta = torch.linalg.pinv(normal + 1e-5 * torch.eye(3)) @ rhs
            errors[held, i] = (j[held, i] @ delta - (target[held, i] - sv[held, i])).norm()
        finite = torch.isfinite(errors[:, i])
        # A held-out view is accepted only when it agrees with a fit built from
        # the other views. Exact-2 tracks intentionally skip this test above.
        accepted[:, i] = vis[:, i] & (~finite | (errors[:, i] <= residual_threshold))
        if int(accepted[:, i].sum()) < min_inliers:
            # Keep the best geometrically supported views rather than silently
            # turning a failed track into an unobserved track.
            order = torch.argsort(torch.nan_to_num(errors[:, i], nan=float("inf")))
            accepted[:, i] = False
            accepted[order[:min_inliers], i] = vis[order[:min_inliers], i]
    return {"accepted_visibility": accepted, "loo_reprojection_error": errors,
            "rejected_visibility": vis & ~accepted,
            "track_consensus_available": vis.sum(0) >= 3,
            "input_visibility": vis}
