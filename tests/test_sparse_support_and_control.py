import torch

from correspondence.sparse_sampling import fixed_views_per_track, observability_report, track_dropout
from correspondence.control_graph import interpolation_weights


def test_track_sampling_and_support_counts():
    visibility = torch.ones(4, 20, dtype=torch.bool)
    fg = torch.ones(20, dtype=torch.bool); fg[-2:] = False
    dropped, selected = track_dropout(visibility, .5, fg, 3)
    report = observability_report(dropped, fg)
    assert int(selected.sum()) == int(dropped.any(0).sum())
    assert report["triangulatable_ge2_fraction"] == report["observed_any_fraction"]
    fixed, _ = fixed_views_per_track(visibility, .5, 2, fg, 3)
    assert int(fixed.sum(0).max()) == 2
    assert torch.equal(fixed[:, ~fg], torch.zeros_like(fixed[:, ~fg]))


def test_control_interpolation_partition_of_unity():
    xyz = torch.randn(50, 3)
    mask = torch.ones(50, dtype=torch.bool); mask[-3:] = False
    controls = torch.tensor([0, 10, 20, 30])
    from correspondence.control_graph import interpolation_weights
    idx, weights = interpolation_weights(xyz, controls, mask, neighbors=3)
    assert torch.allclose(weights[mask].sum(1), torch.ones(int(mask.sum())))
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))
