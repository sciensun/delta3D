"""Silhouette boundary observations kept separate from point target_xy."""
import cv2
import numpy as np


def signed_distance_and_gradient(mask):
    mask = np.asarray(mask).astype(bool)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside - outside
    gx = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return signed.astype(np.float32), np.stack([gx, gy], axis=-1).astype(np.float32)


def sample_silhouette_observation(source_mask, target_mask, source_xy, radius=3.0):
    source_signed, _ = signed_distance_and_gradient(source_mask)
    target_signed, target_gradient = signed_distance_and_gradient(target_mask)
    h, w = source_signed.shape
    x = np.clip(np.rint(source_xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(source_xy[:, 1]).astype(np.int64), 0, h - 1)
    boundary = np.abs(source_signed[y, x]) <= float(radius)
    source_gradient = np.stack([cv2.Sobel(source_signed, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
                                cv2.Sobel(source_signed, cv2.CV_32F, 0, 1, ksize=3) / 8.0], axis=-1)
    return {
        "valid": boundary,
        "source_boundary_distance": source_signed[y, x],
        "source_gradient": source_gradient[y, x],
        "target_signed_distance": target_signed[y, x],
        "target_gradient": target_gradient[y, x],
    }
