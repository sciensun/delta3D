import os
import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


COMMON_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _numbered_key8_index_from_name(image_name):
    elev_match = re.search(r"elev([mp]?)(\d{3})", image_name)
    if elev_match:
        sign = -1 if elev_match.group(1) == "m" else 1
        elevation = sign * int(elev_match.group(2))
        if elevation != 0:
            return None
    match = re.search(r"az(\d{3})", image_name)
    if not match:
        return None
    az = int(match.group(1)) % 360
    if az % 45 != 0:
        return None
    # Manual ChatGPT key8 pack uses 01:az000, 02:az045, ..., 08:az315.
    return az // 45 + 1


def _candidate_names(image_name, view_index=None):
    root, ext = os.path.splitext(image_name)
    names = [image_name]
    if ext:
        names.extend(root + suffix for suffix in COMMON_EXTENSIONS)
    else:
        names.extend(image_name + suffix for suffix in COMMON_EXTENSIONS)
        names.extend(root + "_standard" + suffix for suffix in COMMON_EXTENSIONS)
        if root.startswith("0") and "_" in root:
            parts = root.split("_", 1)
            names.extend(parts[1] + "_standard" + suffix for suffix in COMMON_EXTENSIONS)
    numbered_index = _numbered_key8_index_from_name(image_name)
    if numbered_index is None and view_index is not None and "key8" in image_name.lower():
        numbered_index = int(view_index) + 1
    if numbered_index is not None:
        one_based = numbered_index
        names.extend("{:02d}_standard{}".format(one_based, suffix) for suffix in COMMON_EXTENSIONS)
        names.extend("{:02d}{}".format(one_based, suffix) for suffix in COMMON_EXTENSIONS)
    seen = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def find_style_target_path(viewpoint_cam, style_target_root, split="train", required=True):
    search_dirs = [
        os.path.join(style_target_root, split),
        style_target_root,
        os.path.join(style_target_root, "images"),
    ]
    tried = []
    view_index = getattr(viewpoint_cam, "uid", None)
    for search_dir in search_dirs:
        for name in _candidate_names(viewpoint_cam.image_name, view_index=view_index):
            path = os.path.join(search_dir, name)
            tried.append(path)
            if os.path.isfile(path):
                return path
    if not required:
        return None
    raise FileNotFoundError(
        "Could not find stylized target for view '{}'. Tried:\n{}".format(
            viewpoint_cam.image_name, "\n".join(tried)
        )
    )


def load_style_target_rgba(viewpoint_cam, style_target_root, split="train", device="cuda", required=True):
    path = find_style_target_path(viewpoint_cam, style_target_root, split, required=required)
    if path is None:
        return None
    image = Image.open(path)
    if image.size != (int(viewpoint_cam.image_width), int(viewpoint_cam.image_height)):
        image = image.resize((int(viewpoint_cam.image_width), int(viewpoint_cam.image_height)))
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    image = image.convert("RGBA" if has_alpha else "RGB")
    image = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(device=device)
    return tensor


def load_style_target_image(viewpoint_cam, style_target_root, split="train", device="cuda", required=True):
    rgba = load_style_target_rgba(viewpoint_cam, style_target_root, split, device, required=required)
    if rgba is None:
        return None
    return rgba[:3]


def foreground_mask_from_rgba_or_rgb(image, white_thr=0.97, black_thr=0.03):
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] == 4:
        return image[3:4].clamp(0.0, 1.0)

    rgb = image[:3].clamp(0.0, 1.0)
    near_white = (rgb > white_thr).all(dim=0, keepdim=True)
    near_black = (rgb < black_thr).all(dim=0, keepdim=True)
    return (~(near_white | near_black)).to(dtype=rgb.dtype)


def render_foreground_mask(rendered_rgb, background, threshold=0.05):
    if rendered_rgb.dim() == 4:
        rendered_rgb = rendered_rgb[0]
    bg = background.to(device=rendered_rgb.device, dtype=rendered_rgb.dtype).view(3, 1, 1)
    dist = (rendered_rgb[:3].clamp(0.0, 1.0) - bg).abs().mean(dim=0, keepdim=True)
    return torch.sigmoid((dist - threshold) * 50.0)


def match_image_size(image, reference):
    if image.shape[-2:] == reference.shape[-2:]:
        return image
    warnings.warn("Resizing style target to match the rendered view resolution.", RuntimeWarning)
    return F.interpolate(image.unsqueeze(0), size=reference.shape[-2:], mode="bilinear", align_corners=False)[0]
