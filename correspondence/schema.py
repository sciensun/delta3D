"""Validated source-indexed correspondence bundle.

The target positions are indexed by source Gaussian ID. Missing entries are
represented by ``valid_mask=False`` and are never inferred from zero values.
"""
import json
import os
from dataclasses import dataclass

import torch


@dataclass
class CorrespondenceBundle:
    source_xyz: torch.Tensor
    target_xyz: torch.Tensor
    valid_mask: torch.Tensor
    confidence: torch.Tensor
    support_count: torch.Tensor
    residual_3d: torch.Tensor
    directional_variance: torch.Tensor
    target_xy: object = None
    visibility: object = None
    confidence_2d: object = None
    camera_names: object = None
    anchor_metadata: object = None
    alignment_transform: object = None
    metadata: object = None

    @classmethod
    def from_payload(cls, payload, expected_n=None, device="cpu"):
        if not isinstance(payload, dict):
            raise TypeError("correspondence payload must be a dict")
        source = payload.get("source_xyz")
        target = payload.get("target_xyz")
        if target is None:
            raise ValueError("correspondence bundle requires target_xyz")
        # Older synthetic files stored only {target_xyz, confidence}.  They are
        # accepted for Stage 1's target-only loss, but are marked legacy and do
        # not provide source-index validation for export/alignment.
        if source is None:
            if expected_n is None:
                raise ValueError("legacy correspondence without source_xyz requires expected_n")
            source = torch.zeros((expected_n, 3), device=device)
        else:
            source = torch.as_tensor(source, device=device).float()
        target = torch.as_tensor(target, device=device).float()
        n = source.shape[0]
        if expected_n is not None and n != expected_n:
            raise ValueError("correspondence N={} does not match expected N={}".format(n, expected_n))
        valid = torch.as_tensor(payload.get("valid_mask", payload.get("confidence", torch.ones(n))), device=device).bool().flatten()
        confidence = torch.as_tensor(payload.get("confidence", valid.float()), device=device).float().flatten()
        support = torch.as_tensor(payload.get("support_count", valid.long()), device=device).long().flatten()
        residual = torch.as_tensor(payload.get("residual_3d", torch.zeros(n)), device=device).float().flatten()
        variance = torch.as_tensor(payload.get("directional_variance", torch.zeros(n)), device=device).float().flatten()
        bundle = cls(
            source, target, valid, confidence, support, residual, variance,
            payload.get("target_xy"), payload.get("visibility"), payload.get("confidence_2d"),
            payload.get("camera_names"), payload.get("anchor_metadata"),
            payload.get("alignment_transform"),
            dict(payload.get("metadata", {}), legacy_missing_source_xyz=("source_xyz" not in payload)),
        )
        bundle.validate(expected_n=n)
        return bundle

    @classmethod
    def load(cls, path, expected_n=None, device="cpu"):
        return cls.from_payload(torch.load(path, map_location=device), expected_n=expected_n, device=device)

    def validate(self, expected_n=None, num_views=None):
        n = self.source_xyz.shape[0]
        if expected_n is not None and n != expected_n:
            raise ValueError("bundle Gaussian count mismatch")
        if self.source_xyz.shape != (n, 3) or self.target_xyz.shape != (n, 3):
            raise ValueError("source_xyz and target_xyz must be [N,3]")
        for name, value in (("valid_mask", self.valid_mask), ("confidence", self.confidence),
                            ("support_count", self.support_count), ("residual_3d", self.residual_3d),
                            ("directional_variance", self.directional_variance)):
            if value.shape[0] != n:
                raise ValueError("{} must have length N".format(name))
        if self.target_xy is not None:
            xy = torch.as_tensor(self.target_xy)
            if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] != n:
                raise ValueError("target_xy must have shape [V,N,2]")
            views = xy.shape[0]
            for name, value in (("visibility", self.visibility), ("confidence_2d", self.confidence_2d)):
                if value is not None and torch.as_tensor(value).shape != (views, n):
                    raise ValueError("{} must have shape [V,N]".format(name))
            if num_views is not None and views != num_views:
                raise ValueError("target_xy view count mismatch")
        return self

    def to_payload(self):
        return {
            "source_xyz": self.source_xyz.detach().cpu(), "target_xyz": self.target_xyz.detach().cpu(),
            "valid_mask": self.valid_mask.detach().cpu(), "confidence": self.confidence.detach().cpu(),
            "support_count": self.support_count.detach().cpu(), "residual_3d": self.residual_3d.detach().cpu(),
            "directional_variance": self.directional_variance.detach().cpu(),
            "target_xy": None if self.target_xy is None else torch.as_tensor(self.target_xy).detach().cpu(),
            "visibility": None if self.visibility is None else torch.as_tensor(self.visibility).detach().cpu(),
            "confidence_2d": None if self.confidence_2d is None else torch.as_tensor(self.confidence_2d).detach().cpu(),
            "camera_names": self.camera_names, "anchor_metadata": self.anchor_metadata,
            "alignment_transform": self.alignment_transform, "metadata": self.metadata or {},
        }

    def save(self, path):
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.to_payload(), path)

    def view_index(self, camera_name):
        if self.camera_names is None:
            return None
        names = [os.path.basename(str(x)) for x in self.camera_names]
        base = os.path.basename(str(camera_name))
        return names.index(base) if base in names else None
