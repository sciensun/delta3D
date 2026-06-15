import argparse
import json
from pathlib import Path

import pandas as pd

from utils_download import find_best_3d_file


SOURCE_NAMES = ["source.glb", "source.gltf", "source.zip", "source.fbx", "source.obj"]


def size_mb(path):
    if not path or not Path(path).exists():
        return 0.0
    return round(Path(path).stat().st_size / (1024 * 1024), 4)


def inspect_model_dir(model_dir):
    model_dir = Path(model_dir)
    metadata = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    source_file = ""
    for name in SOURCE_NAMES:
        candidate = model_dir / name
        if candidate.exists():
            source_file = str(candidate)
            break

    best = find_best_3d_file(model_dir, prefer_format="glb")
    extracted_dir = model_dir / "extracted"
    extracted_count = 0
    if extracted_dir.exists():
        extracted_count = sum(1 for path in extracted_dir.rglob("*") if path.is_file())

    fmt = ""
    if best:
        fmt = best.suffix.lower().lstrip(".")
    elif source_file:
        fmt = Path(source_file).suffix.lower().lstrip(".")

    if best:
        status = "ready"
    elif source_file:
        status = "downloaded_no_3d_candidate"
    else:
        status = "missing_source"

    return {
        "model_id": metadata.get("model_id", model_dir.name),
        "model_name": metadata.get("model_name", model_dir.name),
        "local_dir": str(model_dir),
        "source_file": source_file,
        "best_3d_file": str(best) if best else "",
        "format": fmt,
        "size_mb": size_mb(source_file),
        "extracted_file_count": extracted_count,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect downloaded model folders and write a manifest CSV.")
    parser.add_argument("--downloads", default="downloads")
    parser.add_argument("--out", default="outputs/download_manifest.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    downloads = Path(args.downloads)
    rows = []
    if downloads.exists():
        for model_dir in sorted(path for path in downloads.iterdir() if path.is_dir()):
            rows.append(inspect_model_dir(model_dir))
    else:
        print(f"WARNING: downloads folder does not exist: {downloads}")

    pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "model_name",
            "local_dir",
            "source_file",
            "best_3d_file",
            "format",
            "size_mb",
            "extracted_file_count",
            "status",
        ],
    ).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
