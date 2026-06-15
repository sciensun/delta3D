import os

import torch
import torch.nn as nn

from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self):
        return 3 + 3 * 2 * self.num_freqs

    def forward(self, xyz):
        enc = [xyz]
        xb = xyz[..., None, :] * self.freq_bands[:, None]
        enc.extend([torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)])
        return torch.cat(enc, dim=-1)


def _make_mlp(in_dim, out_dim, hidden_dim, depth):
    layers = []
    last_dim = in_dim
    for _ in range(max(depth - 1, 0)):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class StyleDeformModel(nn.Module):
    def __init__(
        self,
        style_names="round",
        style_dim=16,
        num_freqs=6,
        hidden_dim=128,
        depth=4,
        max_d_xyz=0.05,
        max_d_scaling=0.10,
        enable_rotation=False,
    ):
        super().__init__()
        if isinstance(style_names, str):
            style_names = [name.strip() for name in style_names.split(",") if name.strip()]
        if not style_names:
            raise ValueError("StyleDeformModel requires at least one style name.")

        self.style_names = list(style_names)
        self.style_to_idx = {name: idx for idx, name in enumerate(self.style_names)}
        self.style_embedding = nn.Embedding(len(self.style_names), style_dim)
        self.xyz_encoder = FourierPositionalEncoding(num_freqs)
        self.max_d_xyz = max_d_xyz
        self.max_d_scaling = max_d_scaling
        self.enable_rotation = enable_rotation
        self.optimizer = None

        in_dim = self.xyz_encoder.out_dim + style_dim + 1
        delta_dim = 10 if enable_rotation else 6
        self.delta_head = _make_mlp(in_dim, delta_dim, hidden_dim, depth)
        self.mask_head = _make_mlp(in_dim, 1, hidden_dim, depth)

    def _style_indices(self, style_name, n, device):
        if isinstance(style_name, (list, tuple)):
            if len(style_name) != n:
                raise ValueError("style_name list length must match number of Gaussians.")
            indices = [self.style_to_idx[name] for name in style_name]
            return torch.tensor(indices, dtype=torch.long, device=device)
        if style_name not in self.style_to_idx:
            raise ValueError(
                "Unknown style_name '{}'. Available styles: {}".format(style_name, ", ".join(self.style_names))
            )
        return torch.full((n,), self.style_to_idx[style_name], dtype=torch.long, device=device)

    def forward(self, xyz, style_name, alpha):
        if alpha.dim() == 0:
            alpha = alpha.reshape(1, 1).expand(xyz.shape[0], 1)
        elif alpha.dim() == 1:
            alpha = alpha[:, None]
        alpha = alpha.to(device=xyz.device, dtype=xyz.dtype)
        if alpha.shape[0] == 1 and xyz.shape[0] != 1:
            alpha = alpha.expand(xyz.shape[0], 1)
        if alpha.shape != (xyz.shape[0], 1):
            raise ValueError("alpha must be scalar, [N], or [N, 1].")

        style_idx = self._style_indices(style_name, xyz.shape[0], xyz.device)
        style_emb = self.style_embedding(style_idx).to(dtype=xyz.dtype)
        feat = torch.cat([self.xyz_encoder(xyz), style_emb, alpha], dim=-1)

        edit_mask = torch.sigmoid(self.mask_head(feat))
        raw_delta = self.delta_head(feat)
        raw_xyz = torch.tanh(raw_delta[:, :3]) * self.max_d_xyz
        raw_scaling = torch.tanh(raw_delta[:, 3:6]) * self.max_d_scaling

        # Scheme C: time-conditioned deformation is replaced by style-conditioned deformation.
        # The scalar style axis and editability mask gate all geometry deltas.
        gate = alpha * edit_mask
        d_xyz = gate * raw_xyz
        d_scaling = gate * raw_scaling

        if self.enable_rotation:
            d_rotation = gate * torch.tanh(raw_delta[:, 6:10])
        else:
            d_rotation = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype, device=xyz.device)

        return d_xyz, d_rotation, d_scaling, edit_mask

    def step(self, xyz, style_name, alpha):
        return self.forward(xyz, style_name, alpha)

    def train_setting(self, opt):
        lr = getattr(opt, "style_deform_lr", 1e-3)
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.parameters()), "lr": lr, "name": "style_deform"}],
            lr=0.0,
            eps=1e-15,
        )
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=lr,
            lr_final=lr * 0.1,
            lr_delay_mult=getattr(opt, "position_lr_delay_mult", 0.01),
            max_steps=getattr(opt, "deform_lr_max_steps", getattr(opt, "iterations", 40000)),
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "style_deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "style_names": self.style_names,
                "max_d_xyz": self.max_d_xyz,
                "max_d_scaling": self.max_d_scaling,
                "enable_rotation": self.enable_rotation,
            },
            os.path.join(out_weights_path, "style_deform.pth"),
        )

    def load_weights(self, ckpt_path):
        if os.path.isdir(ckpt_path):
            loaded_iter = searchForMaxIteration(os.path.join(ckpt_path, "style_deform"))
            ckpt_path = os.path.join(ckpt_path, "style_deform/iteration_{}/style_deform.pth".format(loaded_iter))
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "style_deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
