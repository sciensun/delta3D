"""Fixed-bank canonical ordinary model: G_ord_canon = G_sty + Delta*."""
import json
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement


class CanonicalOrdinaryModel:
    def __init__(self, source_xyz, d_xyz, d_scaling=None, foreground_mask=None, correspondence_confidence=None, metadata=None):
        self.source_xyz = torch.as_tensor(source_xyz).float().cpu()
        self.d_xyz = torch.as_tensor(d_xyz).float().cpu()
        self.d_scaling = torch.zeros_like(self.d_xyz) if d_scaling is None else torch.as_tensor(d_scaling).float().cpu()
        self.foreground_mask = None if foreground_mask is None else torch.as_tensor(foreground_mask).bool().flatten().cpu()
        self.correspondence_confidence = None if correspondence_confidence is None else torch.as_tensor(correspondence_confidence).float().flatten().cpu()
        self.metadata = metadata or {}
        self.validate()

    @classmethod
    def from_delta(cls, source_xyz, delta_payload, correspondence_bundle=None):
        confidence = delta_payload.get("correspondence_confidence") if isinstance(delta_payload, dict) else None
        if correspondence_bundle is not None: confidence = correspondence_bundle.confidence
        return cls(source_xyz, delta_payload["d_xyz"], delta_payload.get("d_scaling"), delta_payload.get("foreground_mask"), confidence, delta_payload.get("metadata", {}))

    @property
    def num_gaussians(self): return self.source_xyz.shape[0]

    @property
    def canonical_xyz(self): return self.source_xyz + self.d_xyz

    def validate(self):
        n = self.source_xyz.shape[0]
        if self.source_xyz.shape != (n, 3) or self.d_xyz.shape != (n, 3) or self.d_scaling.shape != (n, 3):
            raise ValueError("canonical source/delta tensors must all be [N,3]")
        if self.foreground_mask is not None and self.foreground_mask.shape[0] != n: raise ValueError("foreground mask length mismatch")
        if self.correspondence_confidence is not None and self.correspondence_confidence.shape[0] != n: raise ValueError("confidence length mismatch")
        if not torch.isfinite(self.canonical_xyz).all(): raise ValueError("canonical xyz contains NaN/Inf")
        if not torch.equal(self.d_scaling, torch.zeros_like(self.d_scaling)): raise ValueError("d_scaling must be exactly zero")
        if self.foreground_mask is not None and (self.d_xyz[~self.foreground_mask] != 0).any(): raise ValueError("background delta must be exactly zero")
        return True

    def to_payload(self):
        return {"source_xyz": self.source_xyz, "d_xyz": self.d_xyz, "d_scaling": self.d_scaling, "canonical_xyz": self.canonical_xyz, "foreground_mask": self.foreground_mask, "correspondence_confidence": self.correspondence_confidence, "metadata": self.metadata}

    def save(self, path):
        self.validate(); os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); torch.save(self.to_payload(), path)

    def export_delta(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"d_xyz": self.d_xyz, "d_scaling": self.d_scaling, "source_xyz": self.source_xyz, "foreground_mask": self.foreground_mask, "correspondence_confidence": self.correspondence_confidence, "metadata": self.metadata}, path)

    def export_ply(self, source_ply_path, output_ply_path):
        data = PlyData.read(source_ply_path); vertex = data["vertex"].data.copy()
        if len(vertex) != self.num_gaussians: raise ValueError("source PLY count differs from canonical bank")
        vertex["x"], vertex["y"], vertex["z"] = self.canonical_xyz.numpy().T
        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")], text=data.text).write(output_ply_path)
        return output_ply_path
