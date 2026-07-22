"""Image-first observation schema with legacy CorrespondenceBundle support.

The main route observes generated target images. Target xyz is optional and is
only valid for oracle or hybrid experiments. Validity is always explicit.
"""
import os
from dataclasses import dataclass

import torch


OBSERVATION_MODES = ("observed_2d", "oracle_3d", "hybrid")


@dataclass
class ObservationBundle:
    source_xyz: torch.Tensor
    target_xy: object = None
    visibility_2d: object = None
    confidence_2d: object = None
    support_count_2d: object = None
    candidate_visibility_2d: object = None
    match_error_2d: object = None
    silhouette_observations: object = None
    reprojection_residual: object = None
    target_xyz: object = None
    valid_3d_mask: object = None
    confidence_3d: object = None
    camera_names: object = None
    semantic_region: object = None
    task_metadata: object = None
    metadata: object = None
    observation_mode: str = "observed_2d"

    @classmethod
    def from_payload(cls, payload, expected_n=None, device="cpu", mode=None):
        if not isinstance(payload, dict):
            raise TypeError("observation payload must be a dict")
        source = payload.get("source_xyz")
        target_xyz = payload.get("target_xyz")
        if source is None:
            if expected_n is None:
                raise ValueError("observation bundle requires source_xyz")
            source = torch.zeros((expected_n, 3), device=device)
        else:
            source = torch.as_tensor(source, device=device).float()
        n = source.shape[0]
        if target_xyz is not None:
            target_xyz = torch.as_tensor(target_xyz, device=device).float()
        requested = mode or payload.get("observation_mode")
        if requested is None:
            requested = "hybrid" if target_xyz is not None and payload.get("target_xy") is not None else ("oracle_3d" if target_xyz is not None else "observed_2d")
        if requested not in OBSERVATION_MODES:
            raise ValueError("observation_mode must be one of {}".format(", ".join(OBSERVATION_MODES)))

        # Read both the new names and fields emitted by the old bundle.
        target_xy = payload.get("target_xy")
        visibility = payload.get("visibility_2d", payload.get("visibility"))
        confidence_2d = payload.get("confidence_2d")
        valid_3d = payload.get("valid_3d_mask", payload.get("valid_mask"))
        confidence_3d = payload.get("confidence_3d", payload.get("confidence"))
        if valid_3d is None and target_xyz is not None and confidence_3d is not None:
            valid_3d = confidence_3d
        support_2d = payload.get("support_count_2d", payload.get("support_count"))
        candidate_visibility = payload.get("candidate_visibility_2d")
        match_error = payload.get("match_error_2d")
        residual = payload.get("reprojection_residual", payload.get("residual_3d"))
        if target_xy is not None:
            target_xy = torch.as_tensor(target_xy, device=device).float()
        if visibility is not None:
            visibility = torch.as_tensor(visibility, device=device).bool()
        if confidence_2d is not None:
            confidence_2d = torch.as_tensor(confidence_2d, device=device).float()
        if support_2d is not None:
            support_2d = torch.as_tensor(support_2d, device=device).long().flatten()
        if candidate_visibility is not None:
            candidate_visibility = torch.as_tensor(candidate_visibility, device=device).bool()
        if match_error is not None:
            match_error = torch.as_tensor(match_error, device=device).float()
        if residual is not None:
            residual = torch.as_tensor(residual, device=device).float().flatten()
        if valid_3d is not None:
            valid_3d = torch.as_tensor(valid_3d, device=device).bool().flatten()
        if confidence_3d is not None:
            confidence_3d = torch.as_tensor(confidence_3d, device=device).float().flatten()
        bundle = cls(
            source_xyz=source,
            target_xy=target_xy,
            visibility_2d=visibility,
            confidence_2d=confidence_2d,
            support_count_2d=support_2d,
            candidate_visibility_2d=candidate_visibility,
            match_error_2d=match_error,
            silhouette_observations=payload.get("silhouette_observations"),
            reprojection_residual=residual,
            target_xyz=target_xyz,
            valid_3d_mask=valid_3d,
            confidence_3d=confidence_3d,
            camera_names=payload.get("camera_names"),
            semantic_region=payload.get("semantic_region", payload.get("anchor_metadata")),
            task_metadata=payload.get("task_metadata"),
            metadata=dict(payload.get("metadata", {}), legacy_format=("observation_mode" not in payload)),
            observation_mode=requested,
        )
        return bundle.validate(mode=requested, expected_n=expected_n)

    @classmethod
    def load(cls, path, expected_n=None, device="cpu", mode=None):
        return cls.from_payload(torch.load(path, map_location=device), expected_n=expected_n, device=device, mode=mode)

    def validate(self, mode=None, expected_n=None, num_views=None):
        mode = mode or self.observation_mode
        if mode not in OBSERVATION_MODES:
            raise ValueError("unsupported observation_mode: {}".format(mode))
        if self.source_xyz.ndim != 2 or self.source_xyz.shape[1] != 3:
            raise ValueError("source_xyz must have shape [N,3]")
        n = self.source_xyz.shape[0]
        if expected_n is not None and n != expected_n:
            raise ValueError("observation N={} does not match expected N={}".format(n, expected_n))
        if self.target_xyz is not None and self.target_xyz.shape != (n, 3):
            raise ValueError("target_xyz must have shape [N,3]")
        if mode == "observed_2d" and self.target_xy is None:
            raise ValueError("observed_2d requires target_xy; no oracle projection is allowed")
        if mode in ("oracle_3d", "hybrid") and self.target_xyz is None:
            raise ValueError("{} requires target_xyz".format(mode))
        if self.target_xy is not None:
            if self.target_xy.ndim != 3 or self.target_xy.shape[1:] != (n, 2):
                raise ValueError("target_xy must have shape [V,N,2]")
            views = self.target_xy.shape[0]
            if num_views is not None and views != num_views:
                raise ValueError("target_xy view count mismatch")
            for name, value in (("visibility_2d", self.visibility_2d), ("confidence_2d", self.confidence_2d)):
                if value is not None and value.shape != (views, n):
                    raise ValueError("{} must have shape [V,N]".format(name))
            if self.support_count_2d is not None and self.support_count_2d.shape != (n,):
                raise ValueError("support_count_2d must have length N")
            for name, value in (("candidate_visibility_2d", self.candidate_visibility_2d),
                                ("match_error_2d", self.match_error_2d)):
                if value is not None and value.shape != (views, n):
                    raise ValueError("{} must have shape [V,N]".format(name))
            if self.reprojection_residual is not None and self.reprojection_residual.shape != (n,):
                raise ValueError("reprojection_residual must have length N")
        for name, value in (("valid_3d_mask", self.valid_3d_mask), ("confidence_3d", self.confidence_3d)):
            if value is not None and value.shape != (n,):
                raise ValueError("{} must have length N".format(name))
        if self.camera_names is not None and self.target_xy is not None and len(self.camera_names) != self.target_xy.shape[0]:
            raise ValueError("camera_names length must match target_xy views")
        return self

    def view_index(self, camera_name):
        if self.camera_names is None:
            return None
        names = [os.path.basename(str(x)) for x in self.camera_names]
        base = os.path.basename(str(camera_name))
        return names.index(base) if base in names else None

    def to_payload(self):
        self.validate()
        return {
            "source_xyz": self.source_xyz.detach().cpu(),
            "target_xy": None if self.target_xy is None else self.target_xy.detach().cpu(),
            "visibility_2d": None if self.visibility_2d is None else self.visibility_2d.detach().cpu(),
            "confidence_2d": None if self.confidence_2d is None else self.confidence_2d.detach().cpu(),
            "support_count_2d": None if self.support_count_2d is None else self.support_count_2d.detach().cpu(),
            "candidate_visibility_2d": None if self.candidate_visibility_2d is None else self.candidate_visibility_2d.detach().cpu(),
            "match_error_2d": None if self.match_error_2d is None else self.match_error_2d.detach().cpu(),
            "silhouette_observations": self.silhouette_observations,
            "reprojection_residual": None if self.reprojection_residual is None else self.reprojection_residual.detach().cpu(),
            "target_xyz": None if self.target_xyz is None else self.target_xyz.detach().cpu(),
            "valid_3d_mask": None if self.valid_3d_mask is None else self.valid_3d_mask.detach().cpu(),
            "confidence_3d": None if self.confidence_3d is None else self.confidence_3d.detach().cpu(),
            "camera_names": self.camera_names,
            "semantic_region": self.semantic_region,
            "task_metadata": self.task_metadata,
            "metadata": self.metadata or {},
            "observation_mode": self.observation_mode,
            # Compatibility keys for older scripts/readers.
            "visibility": None if self.visibility_2d is None else self.visibility_2d.detach().cpu(),
            "valid_mask": None if self.valid_3d_mask is None else self.valid_3d_mask.detach().cpu(),
            "confidence": None if self.confidence_3d is None else self.confidence_3d.detach().cpu(),
        }

    def save(self, path):
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.to_payload(), path)

    # Compatibility properties used by existing Stage 1.5 tools.
    @property
    def valid_mask(self):
        return self.valid_3d_mask

    @property
    def confidence(self):
        return self.confidence_3d

    @property
    def visibility(self):
        return self.visibility_2d


# Existing code and old artifacts can continue importing this name.
CorrespondenceBundle = ObservationBundle
