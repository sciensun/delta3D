#!/usr/bin/env python3
"""Render a GLB with true perspective Blender cameras and export Blender/NeRF transforms."""

import argparse
import json
import math
import os
import sys
from datetime import datetime


def blender_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    # Allow: --elevations "-20,0,20" when argparse would otherwise see it as an option.
    fixed = []
    i = 0
    while i < len(argv):
        if argv[i] == "--elevations" and i + 1 < len(argv):
            fixed.append("--elevations=" + argv[i + 1])
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    return fixed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_azimuth", type=int, default=72)
    parser.add_argument("--elevations", default="-20,0,20")
    parser.add_argument("--camera_distance", default="auto")
    parser.add_argument("--fov_degrees", type=float, default=35.0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--transparent_background", action="store_true")
    parser.add_argument("--train_test_split", type=float, default=0.9)
    parser.add_argument("--cycles_samples", type=int, default=64)
    return parser.parse_args(blender_argv())


def parse_elevations(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def clear_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(bpy, input_glb):
    bpy.ops.import_scene.gltf(filepath=input_glb)
    objects = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "EMPTY"}]
    if not objects:
        raise RuntimeError("No renderable objects imported from {}".format(input_glb))
    return objects


def mesh_world_bbox(objects):
    import mathutils

    points = []
    for obj in objects:
        if obj.type == "MESH":
            points.extend(obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box)
    if not points:
        return None
    min_v = mathutils.Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = mathutils.Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v


def normalize_objects(bpy, objects):
    bbox = mesh_world_bbox(objects)
    if bbox is None:
        return None
    min_v, max_v = bbox
    center = (min_v + max_v) * 0.5
    extent_vec = max_v - min_v
    extent = max(extent_vec.x, extent_vec.y, extent_vec.z)
    scale = 1.8 / extent if extent > 0 else 1.0

    root = bpy.data.objects.new("normalized_object_root", None)
    bpy.context.collection.objects.link(root)
    for obj in objects:
        if obj.parent is None:
            obj.parent = root
    root.location = -center * scale
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    norm_min, norm_max = mesh_world_bbox(objects)
    max_dim = max((norm_max - norm_min).x, (norm_max - norm_min).y, (norm_max - norm_min).z)
    return {
        "original_center": [center.x, center.y, center.z],
        "original_extent": extent,
        "scale": scale,
        "normalized_min": [norm_min.x, norm_min.y, norm_min.z],
        "normalized_max": [norm_max.x, norm_max.y, norm_max.z],
        "normalized_max_dim": max_dim,
    }


def setup_scene(bpy, args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.cycles_samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = args.transparent_background
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (1.0, 1.0, 1.0) if args.white_background else (0.0, 0.0, 0.0)

    bpy.ops.object.light_add(type="AREA", location=(0, -3, 4))
    key = bpy.context.object
    key.name = "Key_Area_Light"
    key.data.energy = 500
    key.data.size = 5
    bpy.ops.object.light_add(type="AREA", location=(-3, 3, 3))
    fill = bpy.context.object
    fill.name = "Fill_Area_Light"
    fill.data.energy = 80
    fill.data.size = 6


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(bpy, fov_degrees):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Perspective_Render_Camera"
    camera.data.type = "PERSP"
    camera.data.angle = math.radians(fov_degrees)
    bpy.context.scene.camera = camera
    return camera


def set_camera_view(bpy, camera, azimuth, elevation, distance):
    import mathutils

    az = math.radians(azimuth)
    el = math.radians(elevation)
    camera.location = (
        distance * math.sin(az) * math.cos(el),
        -distance * math.cos(az) * math.cos(el),
        distance * math.sin(el),
    )
    look_at(camera, mathutils.Vector((0.0, 0.0, 0.0)))
    bpy.context.view_layer.update()


def matrix_to_list(matrix):
    return [[float(v) for v in row] for row in matrix]


def main():
    import bpy

    args = parse_args()
    image_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    clear_scene(bpy)
    objects = import_glb(bpy, args.input_glb)
    normalization = normalize_objects(bpy, objects)
    setup_scene(bpy, args)
    camera = setup_camera(bpy, args.fov_degrees)

    fov_rad = math.radians(args.fov_degrees)
    if args.camera_distance == "auto":
        max_dim = normalization["normalized_max_dim"] if normalization else 1.8
        camera_distance = (0.5 * max_dim / math.tan(0.5 * fov_rad)) * 1.35
    else:
        camera_distance = float(args.camera_distance)

    elevations = parse_elevations(args.elevations)
    azimuths = [i * 360.0 / args.num_azimuth for i in range(args.num_azimuth)]
    views = []
    idx = 0
    for elevation in elevations:
        elev_tag = "elevm{:03d}".format(abs(int(round(elevation)))) if elevation < 0 else "elev{:03d}".format(int(round(elevation)))
        for azimuth in azimuths:
            az_tag = "az{:03d}".format(int(round(azimuth)) % 360)
            views.append({"index": idx, "azimuth": azimuth, "elevation": elevation, "name": "{}_{}".format(elev_tag, az_tag)})
            idx += 1

    train_frames = []
    test_frames = []
    view_meta = []
    test_period = max(int(round(1.0 / max(1e-6, 1.0 - args.train_test_split))), 2)

    for view in views:
        set_camera_view(bpy, camera, view["azimuth"], view["elevation"], camera_distance)
        filename = "{:04d}_{}.png".format(view["index"], view["name"])
        filepath = os.path.join(image_dir, filename)
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

        frame = {
            "file_path": "images/" + os.path.splitext(filename)[0],
            "time": view["index"] / max(len(views) - 1, 1),
            "transform_matrix": matrix_to_list(camera.matrix_world),
        }
        if view["index"] % test_period == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
        item = dict(view)
        item.update({"filename": filename, "path": filepath, "split": "test" if view["index"] % test_period == 0 else "train"})
        view_meta.append(item)

    transform_common = {
        "camera_angle_x": float(camera.data.angle_x),
        "camera_angle_y": float(camera.data.angle_y),
        "fl_x": float(camera.data.lens),
        "w": args.resolution,
        "h": args.resolution,
    }
    for name, frames in (("transforms_train.json", train_frames), ("transforms_test.json", test_frames)):
        payload = dict(transform_common)
        payload["frames"] = frames
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    with open(os.path.join(args.out_dir, "view_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"object_id": args.object_id, "views": view_meta}, f, indent=2)
    with open(os.path.join(args.out_dir, "render_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "object_id": args.object_id,
                "input_glb": os.path.abspath(args.input_glb),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "resolution": args.resolution,
                "num_azimuth": args.num_azimuth,
                "elevations": elevations,
                "camera_distance": camera_distance,
                "fov_degrees": args.fov_degrees,
                "camera_angle_x": float(camera.data.angle_x),
                "camera_angle_y": float(camera.data.angle_y),
                "white_background": args.white_background,
                "transparent_background": args.transparent_background,
                "train_test_split": args.train_test_split,
                "normalization": normalization,
                "num_train": len(train_frames),
                "num_test": len(test_frames),
            },
            f,
            indent=2,
        )
    print("Wrote perspective dataset:", args.out_dir)
    print("Images:", image_dir)
    print("Train frames:", len(train_frames), "Test frames:", len(test_frames))


if __name__ == "__main__":
    main()
