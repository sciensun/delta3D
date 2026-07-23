"""Track-aware sampling and observability diagnostics for synthetic bundles."""
import torch


def camera_centers(cameras):
    """Recover centers from the row-vector world-view matrices used here."""
    return torch.stack([torch.linalg.inv(c.world_view_transform)[3, :3].float() for c in cameras])


def pairwise_baseline_scores(cameras):
    centers = camera_centers(cameras)
    return torch.cdist(centers, centers)


def eligible_tracks(visibility, foreground_mask, min_views=2):
    v = torch.as_tensor(visibility).bool(); fg = torch.as_tensor(foreground_mask).bool()
    return fg & (v.sum(0) >= int(min_views))


def _choose_views(valid, k, pair_scores=None, generator=None):
    if len(valid) <= k:
        return valid
    if pair_scores is None:
        return valid[torch.randperm(len(valid), generator=generator)[:k]]
    local = pair_scores[valid][:, valid]
    first, second = torch.nonzero(local == local.max(), as_tuple=True)
    chosen = [int(valid[first[0]]), int(valid[second[0]])]
    while len(chosen) < k:
        remaining = [int(x) for x in valid.tolist() if int(x) not in chosen]
        score = torch.stack([pair_scores[remaining][:, chosen].min(1).values], 0).squeeze(0)
        chosen.append(remaining[int(score.argmax())])
    return torch.tensor(chosen, dtype=torch.long)


def support_histogram(visibility, foreground_mask=None):
    v = torch.as_tensor(visibility).bool()
    mask = torch.ones(v.shape[1], dtype=torch.bool) if foreground_mask is None else torch.as_tensor(foreground_mask).bool()
    counts = v[:, mask].sum(0)
    return {str(k): int((counts == k).sum()) for k in range(0, int(v.shape[0]) + 1)}, counts


def track_dropout(visibility, fraction, foreground_mask, seed):
    v = torch.as_tensor(visibility).bool().clone()
    fg = torch.as_tensor(foreground_mask).bool()
    g = torch.Generator().manual_seed(int(seed))
    keep = (torch.rand(v.shape[1], generator=g) < float(fraction)) & fg
    return v & keep[None], keep


def fixed_views_per_track(visibility, fraction, views_per_track, foreground_mask, seed,
                          baseline_scores=None, min_original_views=None):
    v = torch.as_tensor(visibility).bool().clone()
    fg = torch.as_tensor(foreground_mask).bool()
    g = torch.Generator().manual_seed(int(seed))
    min_original_views = views_per_track if min_original_views is None else min_original_views
    selected = (torch.rand(v.shape[1], generator=g) < float(fraction)) & eligible_tracks(v, fg, min_original_views)
    out = torch.zeros_like(v)
    for i in torch.nonzero(selected).flatten().tolist():
        valid = torch.nonzero(v[:, i]).flatten()
        if len(valid) <= views_per_track:
            out[valid, i] = True
            continue
        chosen = _choose_views(valid, views_per_track, baseline_scores, g)
        out[chosen, i] = True
    return out, selected


def observability_report(visibility, foreground_mask, active_mask=None, baseline_scores=None):
    v = torch.as_tensor(visibility).bool(); fg = torch.as_tensor(foreground_mask).bool()
    active = fg if active_mask is None else torch.as_tensor(active_mask).bool()
    counts = v.sum(0)
    pair = counts >= 2; triple = counts >= 3
    report = {"histogram_foreground": {str(k): int(((counts == k) & fg).sum()) for k in range(v.shape[0] + 1)},
              "histogram_active": {str(k): int(((counts == k) & active).sum()) for k in range(v.shape[0] + 1)},
              "foreground_total": int(fg.sum()), "active_total": int(active.sum()),
              "observed_any_fraction": float((counts[fg] > 0).float().mean()),
              "triangulatable_ge2_fraction": float(pair[fg].float().mean()),
              "triangulatable_ge3_fraction": float(triple[fg].float().mean()),
              "active_triangulatable_ge2_fraction": float(pair[active].float().mean()),
              "active_triangulatable_ge3_fraction": float(triple[active].float().mean()),
              "track_counts": counts.tolist()}
    if baseline_scores is not None:
        b = torch.as_tensor(baseline_scores).float()
        vals = b[v].flatten()
        report["baseline_pair_p50"] = float(vals.median()) if len(vals) else 0.0
        report["baseline_pair_p90"] = float(vals.quantile(.9)) if len(vals) else 0.0
    return report
