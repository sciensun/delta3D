"""Stage 1.5 stable-style-delta payload and confidence composition."""
import os
from dataclasses import dataclass

import torch


@dataclass
class StableStyleDelta:
    source_xyz: torch.Tensor
    stable_d_xyz: torch.Tensor
    confidence: torch.Tensor
    view_consistency: torch.Tensor
    repeat_consistency: torch.Tensor
    structure_consistency: torch.Tensor
    intensity_metadata: dict
    style_task_metadata: dict
    metadata: dict

    def validate(self):
        n = self.source_xyz.shape[0]
        if self.source_xyz.shape != (n, 3) or self.stable_d_xyz.shape != (n, 3):
            raise ValueError("source_xyz and stable_d_xyz must be [N,3]")
        for name in ("confidence", "view_consistency", "repeat_consistency", "structure_consistency"):
            if getattr(self, name).flatten().shape[0] != n:
                raise ValueError("{} must have length N".format(name))
        if not torch.isfinite(self.stable_d_xyz).all():
            raise ValueError("stable_d_xyz contains NaN/Inf")
        return self

    def to_payload(self):
        self.validate()
        return {
            "source_xyz": self.source_xyz.detach().cpu(),
            "stable_d_xyz": self.stable_d_xyz.detach().cpu(),
            "confidence": self.confidence.detach().cpu(),
            "view_consistency": self.view_consistency.detach().cpu(),
            "repeat_consistency": self.repeat_consistency.detach().cpu(),
            "structure_consistency": self.structure_consistency.detach().cpu(),
            "intensity_metadata": self.intensity_metadata,
            "style_task_metadata": self.style_task_metadata,
            "metadata": self.metadata,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        value = cls(**payload)
        return value.validate()


def compose_confidence(view_consistency, repeat_consistency, structure_consistency, weights=None):
    values = [torch.as_tensor(x).float().clamp(0, 1) for x in (view_consistency, repeat_consistency, structure_consistency)]
    weights = weights or (1.0, 1.0, 1.0)
    total = sum(float(x) for x in weights)
    return sum(x * float(w) for x, w in zip(values, weights)) / max(total, 1e-8)
