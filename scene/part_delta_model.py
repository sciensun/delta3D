import os

import torch
import torch.nn as nn


class PartDeltaModel(nn.Module):
    """Part-aware Stage 1 delta: d_i = sum_k weight(i,k) Delta_part[k]."""

    def __init__(
        self,
        xyz,
        part_labels,
        foreground_mask=None,
        num_parts=16,
        max_d_xyz=0.03,
        max_d_scaling=0.0,
        assignment_temperature=0.15,
        disable_d_scaling=True,
        enable_rotation=False,
    ):
        super().__init__()
        device = xyz.device
        self.num_gaussians = xyz.shape[0]
        self.num_parts = int(num_parts)
        self.max_d_xyz = max_d_xyz
        self.max_d_scaling = max_d_scaling
        self.disable_d_scaling = disable_d_scaling
        self.enable_rotation = enable_rotation

        labels = part_labels.to(device=device, dtype=torch.long).flatten()
        if labels.shape[0] != self.num_gaussians:
            raise ValueError("part_labels length does not match Gaussian count.")
        if foreground_mask is None:
            foreground_mask = labels >= 0
        foreground_mask = foreground_mask.to(device=device, dtype=torch.float32).flatten()

        centroids = []
        for part_id in range(self.num_parts):
            part_mask = labels == part_id
            if part_mask.any():
                centroids.append(xyz[part_mask].mean(dim=0))
            else:
                centroids.append(xyz[foreground_mask > 0.5].mean(dim=0))
        centroids = torch.stack(centroids, dim=0)
        dist = torch.cdist(xyz.detach(), centroids.detach()).clamp_min(1e-8)
        weights = torch.softmax(-dist / max(float(assignment_temperature), 1e-6), dim=-1)
        weights = weights * foreground_mask[:, None]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights * foreground_mask[:, None]

        self.register_buffer("part_weights", weights)
        self.register_buffer("foreground_mask", foreground_mask[:, None])
        self.register_buffer("part_labels", labels)
        self.raw_part_d_xyz = nn.Parameter(torch.zeros(self.num_parts, 3, device=device))
        if disable_d_scaling:
            self.register_parameter("raw_part_d_scaling", None)
        else:
            self.raw_part_d_scaling = nn.Parameter(torch.zeros(self.num_parts, 3, device=device))
        if enable_rotation:
            self.raw_part_d_rotation = nn.Parameter(torch.zeros(self.num_parts, 4, device=device))
        else:
            self.register_parameter("raw_part_d_rotation", None)

    def forward(self):
        part_d_xyz = torch.tanh(self.raw_part_d_xyz) * self.max_d_xyz
        d_xyz = self.part_weights @ part_d_xyz
        if self.disable_d_scaling:
            part_d_scaling = torch.zeros((self.num_parts, 3), dtype=d_xyz.dtype, device=d_xyz.device)
            d_scaling = torch.zeros((self.num_gaussians, 3), dtype=d_xyz.dtype, device=d_xyz.device)
        else:
            part_d_scaling = torch.tanh(self.raw_part_d_scaling) * self.max_d_scaling
            d_scaling = self.part_weights @ part_d_scaling
        if self.enable_rotation:
            part_d_rotation = torch.tanh(self.raw_part_d_rotation)
            d_rotation = self.part_weights @ part_d_rotation
        else:
            part_d_rotation = torch.zeros((self.num_parts, 4), dtype=d_xyz.dtype, device=d_xyz.device)
            d_rotation = torch.zeros((self.num_gaussians, 4), dtype=d_xyz.dtype, device=d_xyz.device)
        return d_xyz, d_rotation, d_scaling, part_d_xyz, part_d_scaling, part_d_rotation

    def step(self):
        d_xyz, d_rotation, d_scaling, _, _, _ = self.forward()
        return d_xyz, d_rotation, d_scaling

    def save_delta(self, path, gaussians, metadata=None):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        d_xyz, d_rotation, d_scaling, part_d_xyz, part_d_scaling, part_d_rotation = self.forward()
        payload = {
            "d_xyz": d_xyz.detach().cpu(),
            "d_scaling": d_scaling.detach().cpu(),
            "d_rotation": d_rotation.detach().cpu(),
            "part_d_xyz": part_d_xyz.detach().cpu(),
            "part_d_scaling": part_d_scaling.detach().cpu(),
            "part_d_rotation": part_d_rotation.detach().cpu(),
            "part_weights": self.part_weights.detach().cpu(),
            "part_labels": self.part_labels.detach().cpu(),
            "foreground_mask": self.foreground_mask.detach().cpu().flatten(),
            "source_xyz": gaussians.get_xyz.detach().cpu(),
            "metadata": metadata or {},
        }
        try:
            payload["source_scaling"] = gaussians.get_scaling.detach().cpu()
        except Exception:
            pass
        torch.save(payload, path)
