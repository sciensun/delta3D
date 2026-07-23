import torch

from correspondence.sparse_sampling import (fixed_views_per_track, observability_report,
                                             track_dropout, conditioning_metrics)
from correspondence.control_graph import interpolation_weights
from correspondence.cpu_recovery import solve_symmetric_graph
from correspondence.benchmark_artifacts import upsert_records, validate_records


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


def test_symmetric_global_graph_solver_matches_small_quadratic():
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    weights = torch.ones(3)
    rhs = torch.arange(4, dtype=torch.float32)[:, None]
    solved, info = solve_symmetric_graph(rhs, torch.ones(4), edges, weights,
                                         graph_lambda=0.2)
    assert info["cg_info"] == [0]
    # Constant RHS remains close to constant under a Laplacian; this also
    # checks that each undirected edge is assembled exactly once.
    constant, _ = solve_symmetric_graph(torch.ones(4, 1), torch.ones(4), edges, weights,
                                        graph_lambda=0.2)
    assert torch.allclose(constant, torch.ones(4, 1), atol=1e-4)


def test_benchmark_records_replace_duplicate_keys():
    a = {"teacher": "body", "mode": "track", "fraction": .2, "seed": 1}
    merged = upsert_records([dict(a, value=1)], [dict(a, value=2)])
    assert len(merged) == 1 and merged[0]["value"] == 2
    assert validate_records(merged)["unique"] == 1


def test_conditioning_metrics_and_eligible_k_views():
    class Camera:
        def __init__(self, x):
            self.world_view_transform = torch.eye(4); self.world_view_transform[3, 0] = -x
    cameras = [Camera(0.), Camera(1.), Camera(2.)]
    xyz = torch.zeros(4, 3); visibility = torch.ones(3, 4, dtype=torch.bool)
    metrics = conditioning_metrics(xyz, cameras, visibility)
    assert float(metrics["baseline"].min()) > 0
    assert float(metrics["ray_angle"].max()) >= 0
    sampled, selected = fixed_views_per_track(visibility, 1.0, 2, torch.ones(4, dtype=torch.bool), 1)
    assert torch.equal(sampled.sum(0), torch.full((4,), 2))
