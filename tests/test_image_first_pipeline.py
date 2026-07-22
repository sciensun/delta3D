"""CPU regression tests for the image-first project contracts."""
import torch

from correspondence.schema import ObservationBundle, CorrespondenceBundle
from stage1.outputs import save_stage1_delta
from stage1.reliability import StableStyleDelta
from style_data.schema import StyleTaskRecord


def task_record():
    return StyleTaskRecord(
        object_id="demo", object_category="sculpture", source_glb="source.glb",
        source_3dgs="source_model", style_family="standard", source_attributes={},
        target_attributes={"roundness": 0.8}, intensity=1.0, repeat_id="r01",
        camera_names=["view_000", "view_001"], source_image_root="source_images",
        target_image_root="target_images", generation_prompt="preserve view",
    )


def test_style_record_roundtrip(tmp_path):
    record = task_record()
    path = tmp_path / "task.json"
    record.save_json(path)
    assert StyleTaskRecord.load_json(path).to_dict() == record.to_dict()


def test_observed_2d_without_target_xyz_and_camera_index():
    source = torch.zeros(4, 3)
    bundle = ObservationBundle(
        source_xyz=source,
        target_xy=torch.zeros(2, 4, 2),
        visibility_2d=torch.ones(2, 4, dtype=torch.bool),
        confidence_2d=torch.ones(2, 4),
        support_count_2d=torch.ones(4, dtype=torch.long),
        camera_names=["view_000", "view_001"],
        observation_mode="observed_2d",
    ).validate()
    assert bundle.target_xyz is None
    assert bundle.view_index("view_001") == 1


def test_observation_modes_and_legacy_loader():
    source = torch.zeros(3, 3)
    target = torch.ones(3, 3)
    oracle = ObservationBundle(source_xyz=source, target_xyz=target,
                               valid_3d_mask=torch.ones(3, dtype=torch.bool),
                               confidence_3d=torch.ones(3), observation_mode="oracle_3d")
    oracle.validate()
    hybrid = ObservationBundle(source_xyz=source, target_xyz=target,
                               target_xy=torch.zeros(1, 3, 2),
                               camera_names=["v"], observation_mode="hybrid")
    hybrid.validate()
    legacy = CorrespondenceBundle.from_payload({"source_xyz": source, "target_xyz": target, "confidence": torch.ones(3)})
    assert legacy.observation_mode == "oracle_3d"


def test_observed_2d_requires_target_xy():
    try:
        ObservationBundle(source_xyz=torch.zeros(2, 3), observation_mode="observed_2d").validate()
    except ValueError as exc:
        assert "target_xy" in str(exc)
    else:
        raise AssertionError("observed_2d without target_xy must fail")


def test_empty_valid_masks_are_explicit():
    bundle = ObservationBundle(
        source_xyz=torch.zeros(2, 3), target_xyz=torch.zeros(2, 3),
        valid_3d_mask=torch.zeros(2, dtype=torch.bool), confidence_3d=torch.zeros(2),
        observation_mode="oracle_3d",
    )
    assert bundle.validate().valid_3d_mask.sum() == 0


def test_stage1_and_stable_delta_serialization(tmp_path):
    class FakeGaussians:
        get_xyz = torch.zeros(3, 3)

    mask = torch.tensor([True, True, False])
    path = tmp_path / "delta.pt"
    save_stage1_delta(path, FakeGaussians(), torch.zeros(3, 3), torch.zeros(3, 4), torch.zeros(3, 3),
                      {"observation_mode": "observed_2d"}, {"foreground_mask": mask})
    payload = torch.load(path, map_location="cpu")
    assert torch.equal(payload["d_scaling"], torch.zeros(3, 3))
    assert torch.equal(payload["d_xyz"][~mask], torch.zeros(1, 3))

    stable_path = tmp_path / "stable.pt"
    StableStyleDelta(
        source_xyz=torch.zeros(3, 3), stable_d_xyz=torch.zeros(3, 3),
        confidence=torch.ones(3), view_consistency=torch.ones(3),
        repeat_consistency=torch.ones(3), structure_consistency=torch.ones(3),
        intensity_metadata={"intensity": 1.0}, style_task_metadata=task_record().to_dict(), metadata={},
    ).save(stable_path)
    assert StableStyleDelta.load(stable_path).stable_d_xyz.shape == (3, 3)
