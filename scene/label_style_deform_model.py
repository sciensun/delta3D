import torch
import torch.nn as nn


def make_mlp(in_dim, out_dim, hidden_dim=128, depth=4):
    layers = []
    last = in_dim
    for _ in range(max(depth - 1, 0)):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class LabelStyleDeformModel(nn.Module):
    """Learn B(F_i, delta_z) -> delta_i plus an editability mask."""

    def __init__(self, feature_dim, delta_z_dim, hidden_dim=128, depth=4, max_d_xyz=0.03, max_d_scaling=0.08):
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_z_dim = delta_z_dim
        self.max_d_xyz = max_d_xyz
        self.max_d_scaling = max_d_scaling
        in_dim = feature_dim + delta_z_dim + 1
        self.delta_head = make_mlp(in_dim, 6, hidden_dim=hidden_dim, depth=depth)
        self.mask_head = make_mlp(in_dim, 1, hidden_dim=hidden_dim, depth=depth)

    def forward(self, features, delta_z, alpha=1.0):
        n = features.shape[0]
        if not torch.is_tensor(delta_z):
            delta_z = torch.tensor(delta_z, dtype=features.dtype, device=features.device)
        delta_z = delta_z.to(device=features.device, dtype=features.dtype)
        if delta_z.dim() == 1:
            delta_z = delta_z[None, :].expand(n, -1)

        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=features.dtype, device=features.device)
        alpha = alpha.to(device=features.device, dtype=features.dtype)
        if alpha.dim() == 0:
            alpha = alpha.view(1, 1).expand(n, 1)
        elif alpha.dim() == 1:
            alpha = alpha[:, None]
        if alpha.shape[0] == 1 and n != 1:
            alpha = alpha.expand(n, 1)

        x = torch.cat([features, delta_z, alpha], dim=-1)
        edit_mask = torch.sigmoid(self.mask_head(x))
        raw = self.delta_head(x)
        bounded_xyz = torch.tanh(raw[:, :3]) * self.max_d_xyz
        bounded_scaling = torch.tanh(raw[:, 3:6]) * self.max_d_scaling
        gate = alpha * edit_mask
        d_xyz = gate * bounded_xyz
        d_scaling = gate * bounded_scaling
        return d_xyz, d_scaling, edit_mask

    def save_weights(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "feature_dim": self.feature_dim,
                "delta_z_dim": self.delta_z_dim,
                "max_d_xyz": self.max_d_xyz,
                "max_d_scaling": self.max_d_scaling,
            },
            path,
        )

    def load_weights(self, path):
        payload = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload)
