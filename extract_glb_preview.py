import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def placeholder(path, size, text):
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size - 9, size - 9], outline=(180, 180, 180))
    draw.text((18, size // 2 - 8), text[:32], fill=(80, 80, 80))
    img.save(path)


def render_with_trimesh(mesh_path, out_path, size):
    import trimesh

    mesh = trimesh.load(mesh_path, force="scene")
    png = mesh.save_image(resolution=(size, size), visible=True)
    if not png:
        raise RuntimeError("trimesh returned empty image")
    out_path.write_bytes(png)


def main():
    parser = argparse.ArgumentParser(description="Generate simple previews from downloaded GLB/3D files when rendering is available.")
    parser.add_argument("--manifest", default="outputs/download_manifest.csv")
    parser.add_argument("--out_dir", default="previews_from_glb")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status_csv", default="outputs/glb_preview_status.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_csv)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest).fillna("")
    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="previewing"):
        model_id = str(row.get("model_id", "model"))
        model_name = str(row.get("model_name", model_id))
        mesh_path = row.get("best_3d_file", "") or row.get("source_file", "")
        out_path = out_dir / f"{model_id}_{model_name[:40]}.png"
        out_path = Path(str(out_path).replace("/", "_"))
        out_path = out_dir / out_path.name

        if out_path.exists() and not args.overwrite:
            rows.append({"model_id": model_id, "preview_path": str(out_path), "status": "exists", "reason": ""})
            continue
        if not mesh_path or not Path(mesh_path).exists():
            placeholder(out_path, args.size, "missing model")
            rows.append({"model_id": model_id, "preview_path": str(out_path), "status": "failed", "reason": "missing_model"})
            continue

        try:
            render_with_trimesh(mesh_path, out_path, args.size)
            rows.append({"model_id": model_id, "preview_path": str(out_path), "status": "rendered", "reason": ""})
        except Exception as exc:
            placeholder(out_path, args.size, "render failed")
            print(f"WARNING: preview failed for {mesh_path}: {exc}", file=sys.stderr)
            rows.append({"model_id": model_id, "preview_path": str(out_path), "status": "failed", "reason": str(exc)})

    pd.DataFrame(rows).to_csv(status_path, index=False)
    print(f"Wrote preview status to {status_path}")


if __name__ == "__main__":
    main()
