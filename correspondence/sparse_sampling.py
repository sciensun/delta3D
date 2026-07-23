"""Track-aware sampling and observability diagnostics for synthetic bundles."""
import torch


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
                          baseline_scores=None):
    v = torch.as_tensor(visibility).bool().clone()
    fg = torch.as_tensor(foreground_mask).bool()
    g = torch.Generator().manual_seed(int(seed))
    selected = (torch.rand(v.shape[1], generator=g) < float(fraction)) & fg
    out = torch.zeros_like(v)
    for i in torch.nonzero(selected).flatten().tolist():
        valid = torch.nonzero(v[:, i]).flatten()
        if len(valid) <= views_per_track:
            out[valid, i] = True
            continue
        if baseline_scores is None:
            order = torch.randperm(len(valid), generator=g)
            chosen = valid[order[:views_per_track]]
        else:
            scores = torch.as_tensor(baseline_scores)[valid]
            chosen = valid[torch.argsort(scores, descending=True)[:views_per_track]]
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
