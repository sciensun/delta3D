"""CPU reprojection-only recovery diagnostic for observed_2d bundles."""
import numpy as np
import torch

from .gaussian_visibility import project_points


def recover_xyz_from_observations(source_xyz, cameras, target_xy, visibility,
                                  confidence=None, iterations=4, huber_delta=3.0,
                                  regularization=1e-4, min_support=2,
                                  propagate=True, silhouette_observations=None,
                                  silhouette_weight=1.0, point_weight=1.0):
    source_xyz = torch.as_tensor(source_xyz).float().cpu()
    target_xy = torch.as_tensor(target_xy).float().cpu()
    visibility = torch.as_tensor(visibility).bool().cpu()
    confidence = torch.ones_like(visibility, dtype=torch.float32) if confidence is None else torch.as_tensor(confidence).float().cpu()
    n = source_xyz.shape[0]
    source_views, jacobians = [], []
    eps = 1e-3
    for camera in cameras:
        base, _, _ = project_points(source_xyz, camera.full_proj_transform,
                                    camera.image_width, camera.image_height)
        columns = []
        for axis in range(3):
            shifted = source_xyz.clone()
            shifted[:, axis] += eps
            plus, _, _ = project_points(shifted, camera.full_proj_transform,
                                        camera.image_width, camera.image_height)
            columns.append((plus - base) / eps)
        source_views.append(base)
        jacobians.append(torch.stack(columns, dim=-1))
    source_views = torch.stack(source_views)
    jacobians = torch.stack(jacobians)
    residual = target_xy - source_views
    weights = visibility.float() * confidence.clamp_min(0) * float(point_weight)
    support = visibility.sum(0)
    if silhouette_observations is not None:
        support = support + torch.as_tensor(silhouette_observations["valid"]).bool().sum(0)
    delta = torch.zeros((n, 3), dtype=torch.float32)
    eye = torch.eye(3).expand(n, 3, 3)
    for _ in range(iterations):
        pred = torch.einsum("vnac,nc->vna", jacobians, delta)
        error = torch.linalg.vector_norm(pred - residual, dim=-1)
        robust = torch.where(error <= huber_delta, torch.ones_like(error), huber_delta / error.clamp_min(1e-6))
        w = weights * robust
        # Explicit accumulation keeps this compatible with torch 1.13 and
        # avoids materializing a large dense view-by-Gaussian normal tensor.
        normal = torch.zeros((n, 3, 3))
        rhs = torch.zeros((n, 3))
        for view in range(len(cameras)):
            j = jacobians[view]
            ww = w[view]
            normal += torch.einsum("n,nij,nik->njk", ww, j, j)
            rhs += torch.einsum("n,nij,ni->nj", ww, j, residual[view])
            if silhouette_observations is not None:
                sobs = silhouette_observations
                sv = torch.as_tensor(sobs["valid"][view]).bool()
                sg = torch.as_tensor(sobs["target_gradient"][view]).float()
                normal2d = sg / torch.linalg.vector_norm(sg, dim=-1, keepdim=True).clamp_min(1e-6)
                desired = -torch.as_tensor(sobs["target_signed_distance"][view]).float()
                sw = sv.float() * float(silhouette_weight)
                scalar_j = (j * normal2d[:, :, None]).sum(dim=1)
                silhouette_error = (scalar_j * delta).sum(dim=-1) - desired
                robust_s = torch.where(silhouette_error.abs() <= huber_delta, torch.ones_like(silhouette_error), huber_delta / silhouette_error.abs().clamp_min(1e-6))
                sw = sw * robust_s
                normal += torch.einsum("n,ni,nj->nij", sw, scalar_j, scalar_j)
                rhs += torch.einsum("n,ni,n->ni", sw, scalar_j, desired)
        normal = torch.nan_to_num(normal) + max(regularization, 1e-3) * eye
        rhs = torch.nan_to_num(rhs)
        solved = torch.bmm(torch.linalg.pinv(normal), rhs.unsqueeze(-1)).squeeze(-1)
        delta = torch.where((support >= min_support)[:, None], solved, torch.zeros_like(solved))
    if propagate:
        candidate = (support > 0).numpy()
        known = (support >= min_support).numpy()
        if candidate.any() and known.any() and (~known & candidate).any():
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(source_xyz[known].numpy())
                query = np.flatnonzero(~known & candidate)
                _, idx = tree.query(source_xyz[query].numpy(), k=min(8, int(known.sum())))
                idx = np.atleast_2d(idx)
                known_delta = delta[torch.from_numpy(np.flatnonzero(known))]
                propagated = known_delta[idx].mean(axis=1)
                delta[torch.from_numpy(query)] = propagated.float()
            except Exception:
                pass
    return {"d_xyz": delta, "support_count": support,
            "reprojection_residual": residual, "source_views": source_views,
            "jacobians": jacobians}
