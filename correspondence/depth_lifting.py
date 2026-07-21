"""Lift accepted image pixels to camera/world 3D points."""
import torch


def lift_pixels_to_camera_xy_depth(xy, depth, fx, fy, cx, cy):
    xy = torch.as_tensor(xy).float(); depth = torch.as_tensor(depth).float().flatten()
    z = depth; x = (xy[:, 0] - cx) * z / fx; y = (xy[:, 1] - cy) * z / fy
    return torch.stack([x, y, z], dim=1)


def lift_camera_to_world(points_camera, camera_to_world):
    points = torch.as_tensor(points_camera).float(); ones = torch.ones((points.shape[0], 1))
    return (torch.cat([points, ones], 1) @ torch.as_tensor(camera_to_world).float().T)[:, :3]
