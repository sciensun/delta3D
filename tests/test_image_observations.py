"""CPU tests for image-derived observation extraction contracts."""
import numpy as np
import torch

from correspondence.image_observations import perturb_target
from correspondence.match_filters import build_point_matches, robust_patch_flow
from correspondence.matching_backends import FarnebackMatcher
from correspondence.observation_evaluation import endpoint_metrics
from correspondence.schema import ObservationBundle


def test_farneback_known_translation_and_no_oracle_output():
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[18:42, 20:44] = 220
    target = np.zeros_like(source)
    target[18:42, 25:49] = 220
    field = FarnebackMatcher().match(source, target)
    xy = np.array([[30.0, 30.0], [35.0, 35.0]], dtype=np.float32)
    mask = np.ones((64, 64), dtype=bool)
    observed, valid, confidence, _ = build_point_matches(
        xy, field, mask, mask, np.ones(2, dtype=bool),
        search_radius=1, max_cycle_error=10.0, min_confidence=0.0,
    )
    assert valid.all()
    assert float(np.median(observed[:, 0] - xy[:, 0])) > 2.0
    assert np.all(confidence >= 0)

    bundle = ObservationBundle(
        source_xyz=torch.zeros(2, 3), target_xy=torch.from_numpy(observed[None]),
        visibility_2d=torch.from_numpy(valid[None]),
        confidence_2d=torch.from_numpy(confidence[None]),
        support_count_2d=torch.from_numpy(valid.astype(np.int64)),
        camera_names=["view_0"], observation_mode="observed_2d",
    ).validate()
    payload = bundle.to_payload()
    assert payload["target_xyz"] is None
    assert "d_xyz" not in payload


def test_visibility_mask_rejection_and_footprint_aggregation():
    flow = np.zeros((16, 16, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    assert np.allclose(robust_patch_flow(flow, np.array([[5.0, 5.0]], np.float32)), [[3, 0]])
    field = type("Field", (), {"flow": flow, "cycle_error": np.zeros((16, 16), np.float32)})()
    source_mask = np.ones((16, 16), dtype=bool)
    target_mask = np.zeros((16, 16), dtype=bool)
    target_mask[:, 8:] = True
    observed, valid, _, _ = build_point_matches(
        np.array([[5.0, 5.0], [12.0, 5.0]], dtype=np.float32), field,
        source_mask, target_mask, np.array([True, False]),
        search_radius=0, min_confidence=0.0,
    )
    assert valid.tolist() == [True, False]


def test_metrics_and_deterministic_perturbation():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = perturb_target(a, {"brightness": 10, "contrast": 1.0}, seed=1)
    assert int(b[0, 0, 0]) == 10
    metrics = endpoint_metrics(torch.tensor([[[1.0, 0.0]]]), torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    assert metrics["median_epe"] == 1.0
