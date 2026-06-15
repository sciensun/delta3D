import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_can_write(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def find_glb(glb_dir, model_id):
    direct_dir = glb_dir / str(model_id)
    if direct_dir.exists():
        for name in ["source.glb", "source.gltf"]:
            candidate = direct_dir / name
            if candidate.exists():
                return candidate
        for name in ["scene.glb", "scene.gltf"]:
            candidate = direct_dir / "extracted" / name
            if candidate.exists():
                return candidate
    for suffix in [".glb", ".gltf"]:
        candidate = glb_dir / f"{model_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def path_from_manifest_row(item, glb_dir):
    for column in ["best_3d_file", "source_file", "glb_path", "downloaded_file"]:
        value = str(item.get(column, "")).strip()
        if value and value.lower() != "nan":
            path = Path(value)
            if path.exists() and path.suffix.lower() in {".glb", ".gltf"}:
                return path
    return find_glb(glb_dir, item.get("model_id", ""))


def as_mesh(scene_or_mesh):
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError(f"trimesh unavailable: {exc}") from exc

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [geom for geom in scene_or_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)
    return scene_or_mesh


def normal_variation(mesh):
    normals = np.asarray(mesh.face_normals)
    if len(normals) < 2:
        return 0.0
    center = normals.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm > 0:
        center = center / norm
    return float(np.mean(1.0 - np.clip(normals @ center, -1.0, 1.0)))


def dihedral_stats(mesh):
    try:
        angles = np.asarray(mesh.face_adjacency_angles)
        if angles.size == 0:
            return math.nan, math.nan, math.nan
        return float(np.nanmean(angles)), float(np.nanstd(angles)), float(np.nanpercentile(angles, 90))
    except Exception:
        return math.nan, math.nan, math.nan


def feature_row(model_id, model_name, path):
    import trimesh

    loaded = trimesh.load(path, force="scene")
    mesh = as_mesh(loaded)
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("no mesh geometry found")

    bbox = np.asarray(mesh.extents, dtype=float)
    positive = bbox[bbox > 1e-9]
    aspect = float(positive.max() / positive.min()) if positive.size else math.nan
    dih_mean, dih_std, dih_p90 = dihedral_stats(mesh)
    roughness = float(np.nan_to_num(dih_std, nan=0.0) + normal_variation(mesh))

    return {
        "model_id": model_id,
        "model_name": model_name,
        "glb_path": str(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bbox_aspect_ratio": aspect,
        "surface_area": float(mesh.area),
        "normal_variation": normal_variation(mesh),
        "dihedral_angle_mean": dih_mean,
        "dihedral_angle_std": dih_std,
        "dihedral_angle_p90": dih_p90,
        "roughness_like_score": roughness,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract numerical GLB geometry features for later style analysis.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--glb_dir", default="glbs")
    parser.add_argument("--out", default="outputs/geometry_features.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    ensure_can_write(out_path, args.overwrite)

    try:
        import trimesh  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"trimesh is required for this script: {exc}")

    manifest = pd.read_csv(args.manifest).fillna("")
    glb_dir = Path(args.glb_dir)
    rows = []
    for _, item in tqdm(manifest.iterrows(), total=len(manifest), desc="extracting"):
        model_id = item.get("model_id", "")
        path = path_from_manifest_row(item, glb_dir)
        if path is None:
            print(f"WARNING: no GLB found for {model_id}", file=sys.stderr)
            continue
        try:
            rows.append(feature_row(model_id, item.get("model_name", ""), path))
        except Exception as exc:
            print(f"WARNING: failed to process {path}: {exc}", file=sys.stderr)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} geometry rows to {out_path}")


if __name__ == "__main__":
    main()
