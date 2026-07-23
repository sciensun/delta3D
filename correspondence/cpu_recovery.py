"""CPU reprojection-only recovery diagnostic for observed_2d bundles."""
import numpy as np
import torch
import time

from .gaussian_visibility import project_points


def graph_energy(delta, edge_index, edge_weights):
    d = torch.as_tensor(delta).float(); e = torch.as_tensor(edge_index).long()
    return float((torch.as_tensor(edge_weights).float() * (d[e[:, 0]] - d[e[:, 1]]).square().sum(1)).sum())


def solve_symmetric_graph(rhs, diagonal, edge_index, edge_weights, graph_lambda=0.01,
                         maxiter=200, tol=1e-5):
    """Solve (diag + lambda L) x = rhs with scipy PCG, one xyz column at a time."""
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, cg
    b = torch.as_tensor(rhs).float().cpu(); diag = torch.as_tensor(diagonal).float().cpu()
    e = torch.as_tensor(edge_index).long().cpu().numpy(); w = torch.as_tensor(edge_weights).float().cpu().numpy()
    n = len(diag); degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, e[:, 0], w); np.add.at(degree, e[:, 1], w)
    def matvec(x):
        y = diag.numpy() * x
        diff = x[e[:, 0]] - x[e[:, 1]]
        np.add.at(y, e[:, 0], graph_lambda * w * diff)
        np.add.at(y, e[:, 1], -graph_lambda * w * diff)
        return y
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    out = np.zeros((n, b.shape[1]), dtype=np.float32); info=[]
    for c in range(b.shape[1]):
        sol, code = cg(op, b[:, c].numpy().astype(np.float64), maxiter=maxiter, tol=tol)
        out[:, c] = torch.from_numpy(sol).float(); info.append(int(code))
    return torch.from_numpy(out), {"cg_info": info, "maxiter": maxiter, "tol": tol}


def _project_with_jacobian(xyz, cameras, jacobian_eps=1e-3):
    source_views, jacobians = [], []
    for camera in cameras:
        base, _, _ = project_points(xyz, camera.full_proj_transform,
                                    camera.image_width, camera.image_height)
        columns = []
        for axis in range(3):
            shifted = xyz.clone(); shifted[:, axis] += jacobian_eps
            plus, _, _ = project_points(shifted, camera.full_proj_transform,
                                        camera.image_width, camera.image_height)
            columns.append((plus - base) / jacobian_eps)
        source_views.append(base); jacobians.append(torch.stack(columns, dim=-1))
    return torch.stack(source_views), torch.stack(jacobians)


def build_geometry_cache(source_xyz, cameras, knn=8, jacobian_eps=1e-3):
    """Build reusable source geometry for multiple observed-2d candidates."""
    started = time.perf_counter()
    xyz = torch.as_tensor(source_xyz).float().cpu()
    source_views, jacobians = _project_with_jacobian(xyz, cameras, jacobian_eps)
    try:
        from scipy.spatial import cKDTree
        _, neighbors = cKDTree(xyz.numpy()).query(xyz.numpy(), k=min(knn + 1, len(xyz)))
        neighbors = np.asarray(neighbors)[:, 1:]
    except Exception:
        neighbors = np.tile(np.arange(len(xyz))[:, None], (1, min(knn, len(xyz))))
    neighbors = torch.from_numpy(neighbors).long()
    # Build a symmetric mutual-KNN graph.  The fixed-width representation is
    # padded with self edges, which keeps the batched solver unchanged.
    directed = neighbors.clone()
    edge_sets = [set(int(x) for x in directed[i].tolist()) for i in range(len(xyz))]
    symmetric = []
    for i in range(len(xyz)):
        mutual = [j for j in edge_sets[i] if i in edge_sets[j]]
        union = mutual or list(edge_sets[i])
        symmetric.append(union[:min(knn, len(union))])
    sym = torch.empty_like(neighbors)
    for i, row in enumerate(symmetric):
        row = row + [i] * (neighbors.shape[1] - len(row))
        sym[i] = torch.tensor(row[:neighbors.shape[1]])
    neighbors = sym
    distance = (xyz[neighbors] - xyz[:, None]).norm(dim=-1).clamp_min(1e-6)
    distance[neighbors == torch.arange(len(xyz))[:, None]] = 1.0
    graph_weights = 1.0 / distance
    graph_weights[neighbors == torch.arange(len(xyz))[:, None]] = 0.0
    graph_weights = graph_weights / graph_weights.sum(1, keepdim=True).clamp_min(1e-8)
    edge_pairs = set()
    for i in range(len(xyz)):
        for j in neighbors[i].tolist():
            if i != int(j): edge_pairs.add(tuple(sorted((i, int(j)))))
    edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long)
    edge_distance = (xyz[edge_index[:, 0]] - xyz[edge_index[:, 1]]).norm(dim=1).clamp_min(1e-6)
    edge_weights = 1.0 / edge_distance
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        rows = edge_index[:, 0].numpy(); cols = edge_index[:, 1].numpy()
        graph = coo_matrix((np.ones(len(rows) * 2), (np.r_[rows, cols], np.r_[cols, rows])), shape=(len(xyz), len(xyz))).tocsr()
        components, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        graph_diag = {"components": int(components), "isolated_vertices": int((sizes == 1).sum()),
                      "largest_component": int(sizes.max()), "smallest_component": int(sizes.min())}
    except Exception:
        graph_diag = {"components": None, "isolated_vertices": None}
    return {"source_xyz": xyz, "source_views": source_views,
            "jacobians": jacobians, "neighbors": neighbors, "graph_weights": graph_weights,
            "degree": torch.ones((len(xyz),), dtype=torch.float32),
            "edge_index": edge_index, "edge_weights": edge_weights,
            "graph_diagnostics": graph_diag,
            "cameras": cameras, "jacobian_eps": jacobian_eps,
            "knn": int(neighbors.shape[1]), "graph_type": "symmetric_mutual_knn",
            "cache_build_seconds": time.perf_counter() - started}


def recover_xyz_graph_coupled_cached(cache, target_xy, visibility, confidence=None,
                                    iterations=40, graph_lambda=0.01,
                                    magnitude_lambda=1e-4, min_support=2,
                                    huber_delta=3.0, foreground_mask=None,
                                    frozen_mask=None, clear_unobserved=False,
                                    jacobian_refresh=2):
    """Vectorized graph-coupled recovery using a reusable geometry cache."""
    started = time.perf_counter()
    target_xy = torch.as_tensor(target_xy).float().cpu()
    visibility = torch.as_tensor(visibility).bool().cpu()
    confidence = torch.ones_like(visibility, dtype=torch.float32) if confidence is None else torch.as_tensor(confidence).float().cpu()
    views, jac, neighbors = cache["source_views"], cache["jacobians"], cache["neighbors"]
    n = views.shape[1]
    foreground_mask = torch.ones(n, dtype=torch.bool) if foreground_mask is None else torch.as_tensor(foreground_mask).bool()
    frozen_mask = torch.zeros(n, dtype=torch.bool) if frozen_mask is None else torch.as_tensor(frozen_mask).bool()
    eye = torch.eye(3).expand(n, 3, 3)
    support = visibility.sum(0)
    delta = torch.zeros((n, 3))
    graph_weights = cache.get("graph_weights", torch.ones(neighbors.shape) / neighbors.shape[1])
    history = []
    downweighted = torch.zeros_like(visibility, dtype=torch.bool)
    for step in range(iterations):
        if step == 0 or (jacobian_refresh and step % jacobian_refresh == 0):
            current_views, current_jac = _project_with_jacobian(cache["source_xyz"] + delta, cache["cameras"], cache["jacobian_eps"])
        else:
            current_views, current_jac = current_views, current_jac
        residual = target_xy - current_views
        pred_error = residual.norm(dim=-1)
        robust = torch.where(pred_error <= huber_delta, torch.ones_like(pred_error), huber_delta / pred_error.clamp_min(1e-6))
        downweighted = robust < 1
        weights = visibility.float() * confidence.clamp_min(0) * robust
        normal = torch.einsum("vn,vnij,vnik->njk", weights, current_jac, current_jac)
        rhs = torch.einsum("vn,vnij,vni->nj", weights, current_jac, residual)
        normal = torch.nan_to_num(normal) + magnitude_lambda * eye
        rhs = torch.nan_to_num(rhs)
        neighbor_mean = (delta[neighbors] * graph_weights[..., None]).sum(1)
        lhs = torch.nan_to_num(normal + graph_lambda * eye + 1e-2 * eye)
        # The reprojection residual is linearized around the current delta, so
        # the solve returns an increment.  Treating it as an absolute delta
        # makes IRLS diverge even with complete, noiseless observations.
        rhs_step = rhs + graph_lambda * (neighbor_mean - delta) - magnitude_lambda * delta
        # The explicit diagonal floor keeps all local 3x3 systems positive
        # definite, avoiding a full-bank pseudoinverse allocation.
        step_update = torch.linalg.solve(lhs, rhs_step)
        step_update[frozen_mask | ~foreground_mask] = 0
        delta = delta + step_update
        active = foreground_mask & ~frozen_mask
        history.append({"iteration": step, "reprojection_loss": float((residual[visibility].square()).mean()) if visibility.any() else 0.0,
                        "update_norm": float(step_update[active].norm()) if active.any() else 0.0,
                        "downweighted_fraction": float(downweighted[visibility].float().mean()) if visibility.any() else 0.0})
    if clear_unobserved:
        delta[(support < min_support) & foreground_mask & ~frozen_mask] = 0
    delta[~foreground_mask | frozen_mask] = 0
    final_views, final_jac = _project_with_jacobian(cache["source_xyz"] + delta,
                                                    cache["cameras"],
                                                    cache["jacobian_eps"])
    residual = target_xy - final_views
    return {"d_xyz": delta, "support_count": support,
            "reprojection_residual": residual, "source_views": current_views,
            "jacobians": current_jac, "neighbors": neighbors, "history": history,
            "downweighted": downweighted,
            "cache_build_seconds": cache.get("cache_build_seconds", 0.0),
            "recovery_seconds": time.perf_counter() - started}


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


def recover_xyz_graph_coupled(source_xyz, cameras, target_xy, visibility,
                              confidence=None, iterations=40, graph_lambda=0.01,
                              magnitude_lambda=1e-4, knn=8, min_support=2,
                              huber_delta=3.0):
    """Joint CPU diagnostic with a source-graph relative-motion penalty.

    This is intentionally a small Jacobi-style solver, not a replacement for
    differentiable Stage 1. It validates whether observations contain enough
    3D information before CUDA optimization is attempted.
    """
    cache = build_geometry_cache(source_xyz, cameras, knn=knn)
    return recover_xyz_graph_coupled_cached(cache, target_xy, visibility,
        confidence, iterations=iterations, graph_lambda=graph_lambda,
        magnitude_lambda=magnitude_lambda, min_support=min_support,
        huber_delta=huber_delta)

    # Legacy scalar implementation retained below for source-level reference.
    base = recover_xyz_from_observations(
        source_xyz, cameras, target_xy, visibility, confidence,
        iterations=0, min_support=min_support, propagate=False,
        huber_delta=huber_delta)
    xyz = torch.as_tensor(source_xyz).float().cpu()
    n = xyz.shape[0]
    support = base["support_count"]
    weights = torch.as_tensor(visibility).float().cpu()
    if confidence is not None:
        weights *= torch.as_tensor(confidence).float().cpu().clamp_min(0)
    residual = base["reprojection_residual"]
    jac = base["jacobians"]
    normal = torch.zeros((n, 3, 3))
    rhs = torch.zeros((n, 3))
    for v in range(len(cameras)):
        err = residual[v].norm(dim=-1)
        robust = torch.where(err <= huber_delta, torch.ones_like(err), huber_delta / err.clamp_min(1e-6))
        w = weights[v] * robust
        normal += torch.einsum("n,nij,nik->njk", w, jac[v], jac[v])
        rhs += torch.einsum("n,nij,ni->nj", w, jac[v], residual[v])
    normal += magnitude_lambda * torch.eye(3).expand(n, 3, 3)
    try:
        from scipy.spatial import cKDTree
        _, neighbors = cKDTree(xyz.numpy()).query(xyz.numpy(), k=min(knn + 1, n))
        neighbors = np.asarray(neighbors)[:, 1:]
    except Exception:
        neighbors = np.tile(np.arange(n)[:, None], (1, 1))
    delta = torch.zeros((n, 3))
    degree = (neighbors >= 0).sum(1).astype(np.float32)
    for _ in range(iterations):
        new_delta = torch.zeros_like(delta)
        for i in range(n):
            nb = torch.from_numpy(neighbors[i][neighbors[i] >= 0]).long()
            lhs = normal[i] + graph_lambda * float(degree[i]) * torch.eye(3)
            local_rhs = rhs[i] + graph_lambda * delta[nb].sum(0) if len(nb) else rhs[i]
            new_delta[i] = torch.linalg.solve(lhs + 1e-6 * torch.eye(3), local_rhs)
        delta = new_delta
    delta[support < min_support] = 0
    return {"d_xyz": delta, "support_count": support,
            "reprojection_residual": residual, "source_views": base["source_views"],
            "jacobians": jac, "neighbors": torch.from_numpy(neighbors).long()}
