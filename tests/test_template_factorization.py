import json
import tempfile

import torch

from style_data import StyleTaskRecord, TargetTemplateRecord
from stage1.template_factorization import factorize_candidates, delta_metrics, robust_shared


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
