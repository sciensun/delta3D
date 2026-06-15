import torch
import torch.nn.functional as F


def image_to_mask(img_tensor, alpha_threshold=0.1, white_threshold=0.08):
    """Convert [3,H,W] or [4,H,W] image tensor to a soft foreground mask [1,H,W]."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    if img_tensor.shape[0] == 4:
        return (img_tensor[3:4] > alpha_threshold).to(dtype=img_tensor.dtype)

    rgb = img_tensor[:3].clamp(0.0, 1.0)
    white = torch.ones((3, 1, 1), dtype=rgb.dtype, device=rgb.device)
    dist = (rgb - white).abs().mean(dim=0, keepdim=True)
    return torch.sigmoid((dist - white_threshold) * 50.0)


def mask_iou(mask_a, mask_b, eps=1e-6):
    mask_a = (mask_a > 0.5).to(dtype=torch.float32)
    mask_b = (mask_b > 0.5).to(dtype=torch.float32)
    inter = (mask_a * mask_b).sum()
    union = ((mask_a + mask_b) > 0).to(dtype=torch.float32).sum()
    return inter / (union + eps)


def mask_loss(mask_a, mask_b):
    if mask_a.shape[-2:] != mask_b.shape[-2:]:
        mask_b = F.interpolate(mask_b.unsqueeze(0), size=mask_a.shape[-2:], mode="nearest")[0]
    return F.l1_loss(mask_a, mask_b)
