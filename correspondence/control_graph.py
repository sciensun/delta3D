"""Low-dimensional control-node translation prior for sparse recovery."""
import torch


def farthest_point_indices(xyz, mask, count, seed=0):
    x = torch.as_tensor(xyz).float(); valid = torch.nonzero(torch.as_tensor(mask).bool()).flatten()
    if len(valid) <= count: return valid
    g = torch.Generator().manual_seed(seed)
    chosen = [int(valid[torch.randint(len(valid), (1,), generator=g)])]
    distance = torch.full((len(valid),), float("inf"))
    for _ in range(1, count):
        distance = torch.minimum(distance, (x[valid] - x[chosen[-1]]).norm(dim=1))
        chosen.append(int(valid[torch.argmax(distance)]))
    return torch.tensor(chosen, dtype=torch.long)


def interpolation_weights(xyz, controls, mask, neighbors=4):
    x = torch.as_tensor(xyz).float(); c = x[torch.as_tensor(controls).long()]
    distances = torch.cdist(x, c)
    vals, idx = torch.topk(distances, min(neighbors, c.shape[0]), largest=False)
    weights = 1.0 / vals.clamp_min(1e-5); weights /= weights.sum(1, keepdim=True)
    weights[~torch.as_tensor(mask).bool()] = 0
    return idx.long(), weights.float()


def fit_control_translation(cache, target_xy, visibility, foreground_mask, control_count=64,
                            iterations=8, magnitude_lambda=1e-4, seed=0):
    xyz = cache["source_xyz"]; fg = torch.as_tensor(foreground_mask).bool()
    controls = farthest_point_indices(xyz, fg, control_count, seed)
    idx, interp = interpolation_weights(xyz, controls, fg)
    m = len(controls); u = torch.zeros((m, 3)); history = []
    target_xy = torch.as_tensor(target_xy).float(); visibility = torch.as_tensor(visibility).bool()
    for it in range(iterations):
        current, jac = _current(cache, xyz, u, controls, idx, interp)
        residual = target_xy - current
        normal = torch.eye(3 * m) * magnitude_lambda
        rhs = torch.zeros(3 * m)
        for v in range(target_xy.shape[0]):
            valid = visibility[v] & fg
            w = valid.float()
            j = jac[v]; r = residual[v]
            local_normal = torch.einsum("n,nij,nik->njk", w, j, j)
            local_rhs = torch.einsum("n,nij,ni->nj", w, j, r)
            # Scatter the four-neighbor interpolation stencil in vectorized
            # blocks; this avoids the previous N*M^2 Python loop.
            for a in range(idx.shape[1]):
                ma = idx[:, a]; wa = interp[:, a]
                rhs_view = local_rhs * wa[:, None]
                rhs_view[~fg] = 0
                rhs[0:3*m].view(m, 3).index_add_(0, ma, rhs_view)
                for b in range(idx.shape[1]):
                    mb = idx[:, b]; wb = interp[:, b]
                    blocks = local_normal * (wa * wb)[:, None, None]
                    blocks[~fg] = 0
                    normal.view(m, 3, m, 3).permute(0, 2, 1, 3).index_put_((ma, mb), blocks, accumulate=True)
        step = torch.linalg.solve(normal + 1e-5 * torch.eye(3*m), rhs)
        u += step.reshape(m, 3)
        history.append({"iteration": it, "step_norm": float(step.norm()),
                        "reprojection_loss": float(residual[visibility].square().mean()) if visibility.any() else 0.0})
    dense = torch.zeros_like(xyz)
    dense[fg] = (interp[fg, :, None] * u[idx[fg]]).sum(1)
    return {"d_xyz": dense, "control_indices": controls, "interpolation_indices": idx,
            "interpolation_weights": interp, "control_translations": u, "history": history}


def _current(cache, xyz, u, controls, idx, interp):
    dense = torch.zeros_like(xyz); dense = (interp[:, :, None] * u[idx]).sum(1)
    from .cpu_recovery import _project_with_jacobian
    return _project_with_jacobian(xyz + dense, cache["cameras"], cache["jacobian_eps"])
