"""Minimal CPU camera/source loader for observation diagnostics."""
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class CpuCamera:
    def __init__(self, image_name, width, height, full_proj_transform):
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.full_proj_transform = full_proj_transform


def load_cpu_source_and_cameras(source_path, model_path, load_iteration=30000):
    dataset = Path(source_path)
    transforms = json.loads((dataset / "transforms_train.json").read_text())
    angle_x = float(transforms["camera_angle_x"])
    width = int(transforms.get("w", 1024))
    height = int(transforms.get("h", width))
    cameras = []
    for frame in transforms["frames"]:
        matrix = np.linalg.inv(np.asarray(frame["transform_matrix"], dtype=np.float32))
        R = -matrix[:3, :3].T
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        world = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1)
        proj = getProjectionMatrix(znear=0.01, zfar=100.0,
                                   fovX=angle_x, fovY=angle_x).transpose(0, 1)
        cameras.append(CpuCamera(Path(frame["file_path"]).stem, width, height, world @ proj))
    ply_path = Path(model_path) / "point_cloud" / ("iteration_{}".format(load_iteration)) / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError("canonical source PLY not found: {}".format(ply_path))
    vertex = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    return torch.from_numpy(xyz), cameras
