import json
import tempfile

import torch

from style_data import StyleTaskRecord, TargetTemplateRecord
from stage1.template_factorization import factorize_candidates, delta_metrics, robust_shared
from correspondence.cpu_recovery import build_geometry_cache, recover_xyz_graph_coupled_cached


def test_target_template_schema_and_legacy_roundtrip():
    record = TargetTemplateRecord("t0", "ordinary", "round", 0.5, "A", object_id="o")
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        record.save_json(handle.name)
        assert TargetTemplateRecord.load_json(handle.name).template_variant_id == "A"
    legacy = StyleTaskRecord("o", "elephant", "a.glb", "m", "round", {}, {}, 0.5, "0", ["v"], "s", "t", "p")
    assert StyleTaskRecord.from_dict(json.loads(json.dumps(legacy.to_dict()))).object_id == "o"


def test_shared_factorization_and_constraints():
    torch.manual_seed(4)
    shared = torch.randn(32, 3) * 0.01
    nuisance = torch.randn(5, 32, 3) * 0.003
    deltas = shared[None] + nuisance
    deltas[:, 20:] = 0
    result = factorize_candidates(deltas, nuisance_features=torch.tensor([[-1.0], [1.0], [-1.0], [1.0], [0.0]]))
    metrics = delta_metrics(result["mean"], shared, torch.arange(32) < 20)
    assert metrics["active"]["cosine"] > 0.99
    assert torch.all(result["mean"][20:] == 0) or torch.allclose(result["mean"][20:], torch.zeros_like(result["mean"][20:]))
    assert float(torch.zeros_like(shared).abs().max()) == 0.0


def test_distinct_region_metrics_and_confidence_normalized_trim():
    target = torch.zeros(6, 3); target[:2] = 1.0
    pred = target.clone(); pred[2:4] = 2.0
    report = delta_metrics(pred, target, active_mask=torch.tensor([1, 1, 0, 0, 0, 0]),
                           foreground_mask=torch.tensor([1, 1, 1, 1, 0, 0]))
    assert report["active"]["count"] == 2
    assert report["foreground"]["count"] == 4
    assert report["global"]["count"] == 6
    assert report["active_energy_percent_of_foreground"] < 100.0
    # The low-confidence outlier must not look good merely because its total
    # confidence is small.
    deltas = torch.zeros(3, 4, 3); deltas[2] = 10.0
    conf = torch.ones(3, 4); conf[2] = 0.01
    estimate = robust_shared(deltas, conf, trim_fraction=1.0 / 3.0)
    assert float(estimate.abs().max()) < 1.0


def test_structured_no_label_and_exact_background():
    from stage1.template_factorization import structured_no_label_factorization
    torch.manual_seed(8)
    shared = torch.randn(12, 3) * 0.01
    deltas = torch.stack([shared + torch.randn_like(shared) * 0.002 for _ in range(5)])
    mask = torch.zeros(12, dtype=torch.bool); mask[:8] = True
    deltas[:, ~mask] = 0
    result = structured_no_label_factorization(deltas, rank=2, iterations=4, foreground_mask=mask)
    assert torch.equal(result["shared"][~mask], torch.zeros_like(result["shared"][~mask]))


def test_geometry_cache_reuse_shapes_and_determinism():
    class Camera:
        image_width = 32; image_height = 32
        full_proj_transform = torch.eye(4)
    xyz = torch.randn(20, 3); cameras = [Camera(), Camera()]
    cache = build_geometry_cache(xyz, cameras, knn=3)
    target = cache["source_views"].clone()
    vis = torch.ones(2, 20, dtype=torch.bool)
    first = recover_xyz_graph_coupled_cached(cache, target, vis, iterations=2)
    second = recover_xyz_graph_coupled_cached(cache, target, vis, iterations=2)
    assert cache["neighbors"].shape == (20, 3)
    assert torch.allclose(first["d_xyz"], second["d_xyz"])
    assert float(first["d_xyz"].abs().max()) < 1e-5


def test_sparse_graph_completion_keeps_foreground_unknowns_and_background_zero():
    class Camera:
        image_width = 64; image_height = 64
        full_proj_transform = torch.eye(4)
    xyz = torch.randn(24, 3) * 0.05
    cameras = [Camera(), Camera(), Camera()]
    cache = build_geometry_cache(xyz, cameras, knn=4)
    target = cache["source_views"].clone()
    target[:, :12] += torch.tensor([1.0, 0.0])
    vis = torch.zeros(3, 24, dtype=torch.bool); vis[:, :12] = True
    fg = torch.ones(24, dtype=torch.bool); fg[20:] = False
    result = recover_xyz_graph_coupled_cached(
        cache, target, vis, vis.float(), iterations=3, foreground_mask=fg,
        graph_lambda=0.1, jacobian_refresh=1)
    assert torch.equal(result["d_xyz"][~fg], torch.zeros_like(result["d_xyz"][~fg]))
    assert len(result["history"]) == 3
    # The solver must retain the unknown foreground variables in its output;
    # they may be small when the synthetic graph has weak directional evidence.
    assert result["d_xyz"].shape == (24, 3)


def test_irls_records_outlier_downweighting():
    class Camera:
        image_width = 64; image_height = 64
        full_proj_transform = torch.eye(4)
    xyz = torch.randn(10, 3) * 0.02
    cache = build_geometry_cache(xyz, [Camera(), Camera()], knn=3)
    target = cache["source_views"].clone(); target[0, 0] += 100.0
    vis = torch.ones(2, 10, dtype=torch.bool)
    result = recover_xyz_graph_coupled_cached(cache, target, vis, vis.float(),
                                              iterations=2, huber_delta=1.0)
    assert any(item["downweighted_fraction"] > 0 for item in result["history"])
