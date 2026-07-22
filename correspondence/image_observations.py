"""Image-derived source-indexed observation extraction."""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gaussian_visibility import estimate_projected_visibility, project_points
from .match_filters import build_point_matches
from .matching_backends import FarnebackMatcher
from .schema import ObservationBundle


def load_rgb_mask(path, threshold=0.08):
    array = np.asarray(Image.open(path).convert("RGBA"))
    rgb = array[:, :, :3]
    alpha = array[:, :, 3]
    if np.any(alpha < 255):
        mask = alpha > 16
    else:
        mask = np.linalg.norm(rgb.astype(np.float32) - 255.0, axis=2) > threshold * 255.0
    return rgb, mask


def perturb_target(rgb, spec=None, seed=0):
    if not spec:
        return rgb
    import cv2
    image = rgb.astype(np.float32)
    image = image * float(spec.get("contrast", 1.0)) + float(spec.get("brightness", 0.0))
    if spec.get("blur_sigma", 0) > 0:
        sigma = float(spec["blur_sigma"])
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if spec.get("noise_std", 0) > 0:
        rng = np.random.default_rng(int(seed))
        image += rng.normal(0, float(spec["noise_std"]), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def find_image(root, image_name):
    stem = Path(str(image_name)).stem
    for path in [Path(root) / str(image_name), *(Path(root).glob(stem + ".*"))]:
        if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return str(path)
    raise FileNotFoundError("image '{}' not found under {}".format(image_name, root))


def extract_image_observations(source_xyz, cameras, source_image_root, target_image_root,
                               foreground_mask=None, foreground_mask_path=None,
                               matcher=None, device="cpu", search_radius=2,
                               min_confidence=0.15, max_cycle_error=3.0,
                               target_perturb=None, perturb_seed=0):
    """Extract observed_2d without target geometry.

    `cameras` must expose full_proj_transform, image_width, image_height, and
    image_name. Gaussian visibility is a documented coarse projected z-buffer.
    """
    source_xyz = torch.as_tensor(source_xyz, device=device).float()
    matcher = matcher or FarnebackMatcher()
    if foreground_mask_path is not None:
        foreground_mask = torch.load(foreground_mask_path, map_location="cpu").bool().numpy()
    foreground_mask = None if foreground_mask is None else np.asarray(foreground_mask).astype(bool)
    all_xy, all_vis, all_conf, names = [], [], [], []
    per_view = []
    for camera in cameras:
        source_path = find_image(source_image_root, camera.image_name)
        target_path = find_image(target_image_root, camera.image_name)
        source_rgb, source_mask = load_rgb_mask(source_path)
        target_rgb, target_mask = load_rgb_mask(target_path)
        if target_rgb.shape[:2] != source_rgb.shape[:2]:
            import cv2
            height, width = source_rgb.shape[:2]
            target_rgb = cv2.resize(target_rgb, (width, height), interpolation=cv2.INTER_AREA)
            target_mask = cv2.resize(target_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        target_rgb = perturb_target(target_rgb, target_perturb,
                                    seed=perturb_seed + len(names))
        if target_perturb and target_perturb.get("mask_erode", 0):
            import cv2
            kernel = np.ones((3, 3), np.uint8)
            iterations = abs(int(target_perturb["mask_erode"]))
            target_mask = cv2.erode(target_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool) if int(target_perturb["mask_erode"]) > 0 else cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
        field = matcher.match(source_rgb, target_rgb)
        xy, in_frame, clip_w = project_points(source_xyz, camera.full_proj_transform,
                                               camera.image_width, camera.image_height)
        xy_cpu = xy.detach().cpu().numpy()
        # clip_w is used as a positive-depth proxy when the camera does not
        # expose a world-view z buffer to this CPU-side matcher.
        visible = estimate_projected_visibility(
            xy_cpu, clip_w.detach().cpu().numpy(), source_mask,
            foreground_mask=foreground_mask, bin_size=4,
        ) & in_frame.detach().cpu().numpy()
        target_xy, valid, confidence, cycle = build_point_matches(
            xy_cpu, field, source_mask, target_mask, visible,
            search_radius=search_radius, max_cycle_error=max_cycle_error,
            min_confidence=min_confidence,
        )
        all_xy.append(target_xy)
        all_vis.append(valid)
        all_conf.append(confidence * valid.astype(np.float32))
        names.append(camera.image_name)
        per_view.append({"camera_name": camera.image_name,
                         "valid_count": int(valid.sum()),
                         "in_frame_count": int(in_frame.sum().item()),
                         "backend": field.backend,
                         "mean_cycle_error": float(np.mean(cycle[visible])) if visible.any() else None})
    visibility = torch.from_numpy(np.stack(all_vis)).bool()
    bundle = ObservationBundle(
        source_xyz=source_xyz.detach().cpu(),
        target_xy=torch.from_numpy(np.stack(all_xy)).float(),
        visibility_2d=visibility,
        confidence_2d=torch.from_numpy(np.stack(all_conf)).float(),
        support_count_2d=visibility.sum(dim=0).long(),
        camera_names=names,
        observation_mode="observed_2d",
        metadata={
            "extraction": "image_derived",
            "visibility_method": "projected_foreground_coarse_zbuffer",
            "matcher": matcher.name,
            "matcher_metadata": getattr(matcher, "params", {}),
            "target_xyz_in_optimizer_input": False,
            "per_view": per_view,
        },
    ).validate()
    return bundle
