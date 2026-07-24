"""Track-level multi-view consistency, separate from per-view IRLS kernels."""
import itertools
import torch


def leave_one_view_out_consensus(source_views, jacobians, target_xy, visibility,
                                 residual_threshold=3.0, min_inliers=2,
                                 fallback="keep_best_two"):
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
        if int(accepted[:, i].sum()) < min_inliers and fallback == "keep_best_two":
            order = torch.argsort(torch.nan_to_num(errors[:, i], nan=float("inf")))
            accepted[:, i] = False
            accepted[order[:min_inliers], i] = vis[order[:min_inliers], i]
        elif int(accepted[:, i].sum()) < min_inliers:
            accepted[:, i] = False
    return {"accepted_visibility": accepted, "loo_reprojection_error": errors,
            "rejected_visibility": vis & ~accepted,
            "track_consensus_available": vis.sum(0) >= 3,
            "input_visibility": vis}


def subset_hypothesis_consensus(source_views, jacobians, target_xy, visibility,
                                residual_threshold=3.0, min_inliers=2,
                                fallback="drop_track", margin_threshold=0.1,
                                confidence_scale=0.1):
    """Select a geometrically consistent subset from K-view tracks.

    Every pair is a hypothesis for exact-3/4 tracks. The hypothesis is scored
    against *all* available views, so this is distinct from leave-one-out.
    No target XYZ or injected outlier labels are used. ``downgrade_confidence``
    keeps ambiguous tracks with a small uniform weight for downstream use.
    """
    sv=torch.as_tensor(source_views).float(); j=torch.as_tensor(jacobians).float()
    target=torch.as_tensor(target_xy).float(); vis=torch.as_tensor(visibility).bool()
    vcount,n=vis.shape; accepted=torch.zeros_like(vis); track_conf=torch.zeros(n); scores=[]; ambiguous=torch.zeros(n,dtype=torch.bool)
    for i in range(n):
        valid=torch.nonzero(vis[:,i]).flatten().tolist()
        if len(valid)<3:
            if len(valid)>=min_inliers and fallback in ("keep_best_two","downgrade_confidence"):
                accepted[valid,i]=True; track_conf[i]=confidence_scale if fallback=="downgrade_confidence" else 1.0
            continue
        candidates=[]
        for subset in itertools.combinations(valid,2):
            normal=torch.zeros((3,3)); rhs=torch.zeros(3)
            for v in subset:
                normal += j[v,i].T @ j[v,i]
                rhs += j[v,i].T @ (target[v,i]-sv[v,i])
            cond=float(torch.linalg.cond(normal + 1e-5*torch.eye(3)))
            if not torch.isfinite(torch.tensor(cond)) or cond>1e8: continue
            d=torch.linalg.solve(normal+1e-5*torch.eye(3),rhs)
            err=torch.stack([(j[v,i]@d-(target[v,i]-sv[v,i])).norm() for v in valid])
            inlier=err<=residual_threshold
            count=int(inlier.sum()); robust=float(torch.where(err<=residual_threshold,err,torch.full_like(err,residual_threshold)).sum())
            candidates.append((count,-robust,-cond,subset,inlier,err))
        if not candidates: continue
        candidates.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
        best=candidates[0]; second=candidates[1] if len(candidates)>1 else None
        score_gap=1.0 if second is None else float(best[0]-second[0])/(max(1,len(valid)))
        ok=best[0]>=min_inliers and (second is None or score_gap>=margin_threshold)
        if ok:
            accepted[torch.tensor(valid)[best[4]],i]=True; track_conf[i]=1.0
        elif fallback=="keep_best_two":
            accepted[list(best[3]),i]=True; track_conf[i]=1.0
        elif fallback=="downgrade_confidence":
            accepted[:,i]=vis[:,i]; track_conf[i]=confidence_scale; ambiguous[i]=True
        # drop_track leaves the complete track rejected.
        scores.append({'gaussian':i,'views':valid,'best_subset':list(best[3]),'inliers':int(best[0]),'second_inliers':None if second is None else int(second[0]),'score_gap':score_gap,'accepted':bool(ok),'ambiguous':bool(not ok)})
    return {'accepted_visibility':accepted,'track_confidence':track_conf,
            'hypotheses':scores,'ambiguous_tracks':ambiguous,
            'rejected_visibility':vis & ~accepted,'input_visibility':vis,
            'mode':'subset_hypothesis'}
