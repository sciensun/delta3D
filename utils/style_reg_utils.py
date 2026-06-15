import torch


def deformation_l2_reg(d_xyz, d_scaling=None, edit_mask=None):
    reg = (d_xyz ** 2).mean()
    if d_scaling is not None and torch.is_tensor(d_scaling):
        reg = reg + (d_scaling ** 2).mean()
    if edit_mask is not None:
        reg = reg + edit_mask.mean()
    return reg


def identity_deform_reg(style_deform, xyz, style_name):
    alpha0 = torch.zeros((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    d_xyz, d_rotation, d_scaling, edit_mask = style_deform.step(xyz, style_name=style_name, alpha=alpha0)
    reg = (d_xyz ** 2).mean() + (d_scaling ** 2).mean()
    if torch.is_tensor(d_rotation):
        reg = reg + (d_rotation ** 2).mean()
    return reg
