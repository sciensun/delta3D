import os

import torch
import torch.nn as nn


class FreeDeltaModel(nn.Module):
    """Per-Gaussian free deformation parameters for one fixed source 3DGS."""

    def __init__(
        self,
        num_gaussians,
        max_d_xyz=0.03,
        max_d_scaling=0.08,
        enable_rotation=False,
        disable_d_scaling=False,
        device="cuda",
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.max_d_xyz = max_d_xyz
        self.max_d_scaling = max_d_scaling
        self.enable_rotation = enable_rotation
        self.disable_d_scaling = disable_d_scaling
        self.raw_d_xyz = nn.Parameter(torch.zeros(num_gaussians, 3, device=device))
        if disable_d_scaling:
            self.register_parameter("raw_d_scaling", None)
        else:
            self.raw_d_scaling = nn.Parameter(torch.zeros(num_gaussians, 3, device=device))
        if enable_rotation:
            self.raw_d_rotation = nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
        else:
            self.register_parameter("raw_d_rotation", None)

    def forward(self):
        d_xyz = torch.tanh(self.raw_d_xyz) * self.max_d_xyz
        if self.disable_d_scaling:
            d_scaling = torch.zeros((self.num_gaussians, 3), dtype=d_xyz.dtype, device=d_xyz.device)
        else:
            d_scaling = torch.tanh(self.raw_d_scaling) * self.max_d_scaling
        if self.enable_rotation:
            d_rotation = torch.tanh(self.raw_d_rotation)
        else:
            d_rotation = torch.zeros((self.num_gaussians, 4), dtype=d_xyz.dtype, device=d_xyz.device)
        return d_xyz, d_rotation, d_scaling

    def step(self):
        return self.forward()

    def save_delta(self, path, gaussians, metadata=None):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        d_xyz, d_rotation, d_scaling = self.step()
        payload = {
            "d_xyz": d_xyz.detach().cpu(),
            "d_scaling": d_scaling.detach().cpu(),
            "d_rotation": d_rotation.detach().cpu(),
            "source_xyz": gaussians.get_xyz.detach().cpu(),
            "metadata": metadata or {},
        }
        try:
            payload["source_scaling"] = gaussians.get_scaling.detach().cpu()
        except Exception:
            pass
        torch.save(payload, path)

    @staticmethod
    def load_delta(path, map_location="cuda"):
        return torch.load(path, map_location=map_location)
