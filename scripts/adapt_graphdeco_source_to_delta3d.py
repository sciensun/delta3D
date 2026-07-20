#!/usr/bin/env python3
"""Adapt a Graphdeco 3DGS output folder to this repo's model layout."""

import argparse
import json
import os
import shutil
from datetime import datetime


REQUIRED_FIELDS = {
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphdeco_model_path", required=True)
    parser.add_argument("--graphdeco_iteration", type=int, default=30000)
    parser.add_argument("--delta3d_model_path", required=True)
    parser.add_argument("--source_dataset_path", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    parser.add_argument("--force", action="store_true", help="Proceed despite schema warnings.")
    return parser.parse_args()


def read_ply_header(path):
    fields = []
    vertex_count = None
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                fields.append(line.split()[-1])
            elif line == "end_header":
                break
    return fields, vertex_count


def link_or_copy(src, dst, copy=False, force=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if not force:
            raise FileExistsError("{} already exists; use --force to replace.".format(dst))
        if os.path.isdir(dst) and not os.path.islink(dst):
            raise IsADirectoryError(dst)
        os.unlink(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def main():
    args = parse_args()
    src_ply = os.path.join(
        args.graphdeco_model_path,
        "point_cloud",
        "iteration_{}".format(args.graphdeco_iteration),
        "point_cloud.ply",
    )
    dst_ply = os.path.join(
        args.delta3d_model_path,
        "point_cloud",
        "iteration_{}".format(args.graphdeco_iteration),
        "point_cloud.ply",
    )
    if not os.path.isfile(src_ply):
        raise FileNotFoundError("Missing Graphdeco PLY: {}".format(src_ply))

    fields, vertex_count = read_ply_header(src_ply)
    field_set = set(fields)
    missing = sorted(REQUIRED_FIELDS - field_set)
    f_rest = sorted(name for name in fields if name.startswith("f_rest_"))

    print("Graphdeco PLY:", src_ply)
    print("vertex count:", vertex_count)
    print("fields:")
    for field in fields:
        print("  " + field)
    if f_rest:
        print("detected f_rest_* fields:", len(f_rest))
    if missing:
        print("WARNING: PLY appears incompatible. Missing fields:")
        for field in missing:
            print("  " + field)
        if not args.force:
            raise SystemExit("Refusing to adapt incompatible PLY without --force.")

    link_or_copy(src_ply, dst_ply, copy=args.copy, force=args.force)
    metadata = {
        "source_builder": "graphdeco",
        "graphdeco_model_path": os.path.abspath(args.graphdeco_model_path),
        "graphdeco_iteration": args.graphdeco_iteration,
        "source_dataset_path": os.path.abspath(args.source_dataset_path),
        "adapted_at": datetime.utcnow().isoformat() + "Z",
        "ply_path": os.path.abspath(dst_ply),
        "copied": bool(args.copy),
        "vertex_count": vertex_count,
        "detected_fields": fields,
        "missing_required_fields": missing,
    }
    os.makedirs(args.delta3d_model_path, exist_ok=True)
    with open(os.path.join(args.delta3d_model_path, "source_builder_graphdeco.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.delta3d_model_path, "README_adapted_source.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Adapted Graphdeco Source\n\n"
            "This folder contains a Graphdeco official 3DGS source PLY arranged in the "
            "local delta3D model layout for Stage 1 delta mining.\n\n"
            "The PLY contents were not modified. If rendering fails or appears blurry, "
            "verify schema compatibility and camera/background settings before Stage 1.\n"
        )
    print("Adapted source written to:", args.delta3d_model_path)
    print("Local PLY:", dst_ply)


if __name__ == "__main__":
    main()
