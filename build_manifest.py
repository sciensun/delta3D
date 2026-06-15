import argparse
import sys
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_name(value):
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def find_column(columns, aliases):
    normalized = {normalize_name(col): col for col in columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def first_present(row, columns):
    for column in columns:
        if column and column in row and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value and value.lower() != "nan":
                return value
    return ""


def ensure_can_write(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def read_images(image_dir):
    images = [
        path
        for path in sorted(Path(image_dir).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        print(f"WARNING: no images found in {image_dir}", file=sys.stderr)
    return images


def build_manifest(image_dir, excel_path, match_mode):
    images = read_images(image_dir)
    df = pd.read_excel(excel_path, engine="openpyxl").fillna("")

    name_col = find_column(df.columns, ["model_name", "name", "title", "model", "model title"])
    url_col = find_column(df.columns, ["source_url", "url", "link", "page_url", "model_url"])
    glb_col = find_column(df.columns, ["glb_url", "glb", "download_url", "download", "direct_glb"])
    sketchfab_col = find_column(df.columns, ["sketchfab_url", "sketchfab", "sketchfab_url", "sketchfab link"])
    notes_col = find_column(df.columns, ["notes", "note", "comment", "comments"])

    records = []
    used_rows = set()

    if match_mode == "order":
        count = min(len(images), len(df))
        if len(images) > len(df):
            for image in images[len(df):]:
                print(f"WARNING: unmatched image by order: {image}", file=sys.stderr)
        if len(df) > len(images):
            for idx in range(len(images), len(df)):
                print(f"WARNING: unmatched Excel row by order: {idx + 2}", file=sys.stderr)

        pairs = [(images[idx], idx) for idx in range(count)]
    else:
        stem_to_images = {}
        for image in images:
            stem_to_images.setdefault(normalize_name(image.stem), []).append(image)

        pairs = []
        for idx, row in df.iterrows():
            candidate = first_present(row, [name_col])
            key = normalize_name(candidate)
            matches = stem_to_images.get(key, [])
            if matches:
                pairs.append((matches.pop(0), idx))
                used_rows.add(idx)
            else:
                print(f"WARNING: unmatched Excel row by stem: {idx + 2} ({candidate})", file=sys.stderr)

        for unmatched in stem_to_images.values():
            for image in unmatched:
                print(f"WARNING: unmatched image by stem: {image}", file=sys.stderr)

    for seq, (image, idx) in enumerate(pairs, 1):
        row = df.iloc[idx]
        model_name = first_present(row, [name_col]) or image.stem
        source_url = first_present(row, [url_col, sketchfab_col, glb_col])
        glb_url = first_present(row, [glb_col])
        sketchfab_url = first_present(row, [sketchfab_col])
        if not sketchfab_url and "sketchfab.com" in source_url.lower():
            sketchfab_url = source_url

        records.append(
            {
                "model_id": f"model_{seq:04d}",
                "model_name": model_name,
                "preview_path": str(image),
                "source_url": source_url,
                "glb_url": glb_url,
                "sketchfab_url": sketchfab_url,
                "notes": first_present(row, [notes_col]),
            }
        )
        used_rows.add(idx)

    return pd.DataFrame(
        records,
        columns=[
            "model_id",
            "model_name",
            "preview_path",
            "source_url",
            "glb_url",
            "sketchfab_url",
            "notes",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Build a preview/model manifest from images and Excel rows.")
    parser.add_argument("--image_dir", default="previews")
    parser.add_argument("--excel", default="models.xlsx")
    parser.add_argument("--out", default="outputs/manifest.csv")
    parser.add_argument("--match_mode", choices=["stem", "order"], default="order")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    ensure_can_write(out_path, args.overwrite)
    manifest = build_manifest(Path(args.image_dir), Path(args.excel), args.match_mode)
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {len(manifest)} rows to {out_path}")


if __name__ == "__main__":
    main()
