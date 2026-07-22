"""Small CPU regression tests for the fixed-bank correspondence interfaces."""
import torch

from correspondence.alignment import apply_similarity, fit_similarity_from_corresponded_points
from correspondence.multiview_fusion import fuse_target_candidates
from correspondence.schema import CorrespondenceBundle


def test_similarity_and_bundle_roundtrip(tmp_path):
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.0 * (source @ rotation.T) + torch.tensor([3.0, -2.0, 1.0])
    transform = fit_similarity_from_corresponded_points(source, target)
    assert torch.allclose(apply_similarity(source, transform), target, atol=1e-5)

    candidates = torch.stack([target, target + 0.01 * torch.randn_like(target)])
    fused = fuse_target_candidates(candidates, torch.ones(2, 4), torch.ones(2, 4))
    bundle = CorrespondenceBundle(
        source_xyz=source,
        target_xyz=fused["target_xyz"],
        valid_3d_mask=fused["valid_mask"],
        confidence_3d=fused["confidence"],
        target_xy=torch.zeros(2, 4, 2),
        visibility_2d=torch.ones(2, 4),
        confidence_2d=torch.ones(2, 4),
        camera_names=["a", "b"],
        observation_mode="hybrid",
    )
    path = tmp_path / "bundle.pt"
    bundle.save(path)
    loaded = CorrespondenceBundle.load(path, expected_n=4)
    assert loaded.valid_mask.all()
    assert loaded.target_xy.shape == (2, 4, 2)
