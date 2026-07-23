import json
import tempfile

import torch

from style_data import StyleTaskRecord, TargetTemplateRecord
from stage1.template_factorization import factorize_candidates, delta_metrics


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
    assert metrics["global_cosine"] > 0.99
    assert torch.all(result["mean"][20:] == 0) or torch.allclose(result["mean"][20:], torch.zeros_like(result["mean"][20:]))
    assert float(torch.zeros_like(shared).abs().max()) == 0.0
