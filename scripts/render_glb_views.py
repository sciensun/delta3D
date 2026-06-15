#!/usr/bin/env python3
"""Render normalized multi-view PNGs from a GLB with Blender.

Run through Blender, for example:
  blender -b --python scripts/render_glb_views.py -- --input_glb assets/3D/foo.glb --out_dir out/key8 --mode key8
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime


DEFAULT_GLB = "/home/shichang/Deformable-3D-Gaussians/assets/3D/big_carved_wooden_elephant_sculpture.glb"
DEFAULT_OBJECT_ID = "big_carved_wooden_elephant_sculpture"
DEFAULT_REPO_ROOT = "/home/shichang/Deformable-3D-Gaussians"


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glb", default=DEFAULT_GLB)
    parser.add_argument("--object_id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--out_dir", default=os.path.join(DEFAULT_REPO_ROOT, "assets/prepared", DEFAULT_OBJECT_ID, "renders_original/key8"))
    parser.add_argument("--mode", choices=("full36", "key8", "tripo"), default="key8")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--ortho_scale", type=float, default=2.45)
    parser.add_argument("--camera_distance", type=float, default=4.0)
    return parser.parse_args(argv)


def view_definitions(mode):
    if mode == "full36":
        views = []
        views.extend({"name": "elev000_az%03d" % az, "azimuth": az, "elevation": 0.0, "hint": "level view"} for az in range(0, 360, 15))
        views.extend({"name": "elevp25_az%03d" % az, "azimuth": az, "elevation": 25.0, "hint": "upper view"} for az in range(0, 360, 45))
        views.extend({"name": "elevm20_az%03d" % az, "azimuth": az, "elevation": -20.0, "hint": "lower view"} for az in range(0, 360, 90))
        return views
    if mode == "key8":
        return [{"name": "key8_az%03d" % az, "azimuth": az, "elevation": 0.0, "hint": view_hint(az)} for az in range(0, 360, 45)]
    return [
        {"name": "tripo_front_3quarter", "azimuth": 35.0, "elevation": 10.0, "hint": "front three-quarter view"},
        {"name": "tripo_front", "azimuth": 0.0, "elevation": 5.0, "hint": "front view"},
        {"name": "tripo_side", "azimuth": 90.0, "elevation": 5.0, "hint": "side view"},
        {"name": "tripo_back_3quarter", "azimuth": 145.0, "elevation": 10.0, "hint": "back three-quarter view"},
    ]


def view_hint(azimuth):
    azimuth = azimuth % 360
    if azimuth in (0, 360):
        return "front view"
    if azimuth in (45, 315):
        return "front three-quarter view"
    if azimuth in (90, 270):
        return "side view"
    if azimuth in (135, 225):
        return "back three-quarter view"
    if azimuth == 180:
        return "back view"
    return "object view"


def clear_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(bpy, input_glb):
    bpy.ops.import_scene.gltf(filepath=input_glb)
    objects = [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "EMPTY"}]
    if not objects:
        raise RuntimeError("No renderable objects imported from %s" % input_glb)
    return objects


def mesh_world_bbox(objects):
    import mathutils

    bbox_points = []
    for obj in objects:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                bbox_points.append(obj.matrix_world @ mathutils.Vector(corner))
    if not bbox_points:
        return None
    min_v = mathutils.Vector((min(p.x for p in bbox_points), min(p.y for p in bbox_points), min(p.z for p in bbox_points)))
    max_v = mathutils.Vector((max(p.x for p in bbox_points), max(p.y for p in bbox_points), max(p.z for p in bbox_points)))
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

    # Keep the post-scale bounding-box center at the world origin. The translation
    # must be scaled as well; otherwise large/small GLBs can render off-center.
    root.location = -center * scale
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    normalized_bbox = mesh_world_bbox(objects)
    if normalized_bbox is None:
        return None
    norm_min, norm_max = normalized_bbox
    return {
        "original_center": [center.x, center.y, center.z],
        "original_extent": extent,
        "scale": scale,
        "normalized_min": [norm_min.x, norm_min.y, norm_min.z],
        "normalized_max": [norm_max.x, norm_max.y, norm_max.z],
    }


def setup_scene(bpy, resolution):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0, -3, 4))
    key = bpy.context.object
    key.name = "Key_Area_Light"
    key.data.energy = 450
    key.data.size = 5
    bpy.ops.object.light_add(type="POINT", location=(-3, 3, 3))
    fill = bpy.context.object
    fill.name = "Soft_Fill_Light"
    fill.data.energy = 80


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(bpy, ortho_scale):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Render_Camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
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


def main():
    import bpy

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clear_scene(bpy)
    objects = import_glb(bpy, args.input_glb)
    normalization = normalize_objects(bpy, objects)
    setup_scene(bpy, args.resolution)
    camera = setup_camera(bpy, args.ortho_scale)

    meta = {
        "object_id": args.object_id,
        "input_glb": os.path.abspath(args.input_glb),
        "mode": args.mode,
        "resolution": args.resolution,
        "normalization": normalization,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "views": [],
    }

    for idx, view in enumerate(view_definitions(args.mode)):
        set_camera_view(bpy, camera, view["azimuth"], view["elevation"], args.camera_distance)
        filename = "%03d_%s.png" % (idx, view["name"])
        out_path = os.path.join(args.out_dir, filename)
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        item = dict(view)
        item.update({"index": idx, "filename": filename, "path": out_path})
        meta["views"].append(item)

    with open(os.path.join(args.out_dir, "views_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
