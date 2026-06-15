import argparse
import math
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def ensure_can_write(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_labels(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, engine="openpyxl").fillna("")
    return pd.read_csv(path).fillna("")


def short_text(value, width):
    value = str(value)
    return value if len(value) <= width else value[: max(0, width - 1)] + "..."


def draw_wrapped(draw, position, text, font, fill, max_chars, line_height, max_lines):
    x, y = position
    lines = textwrap.wrap(str(text), width=max_chars)[:max_lines]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def main():
    parser = argparse.ArgumentParser(description="Create a readable image contact sheet with candidate labels.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--labels", default="outputs/labels_auto.xlsx")
    parser.add_argument("--out", default="outputs/contact_sheet.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tile_image_size", type=int, default=170)
    parser.add_argument("--cols", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    ensure_can_write(out_path, args.overwrite)

    manifest = pd.read_csv(args.manifest).fillna("")
    labels = load_labels(args.labels)
    data = manifest.merge(labels, on=["model_id"], how="left", suffixes=("", "_label"))

    image_size = args.tile_image_size
    tile_w = image_size + 40
    tile_h = image_size + 96
    cols = args.cols or max(1, min(5, math.ceil(math.sqrt(max(1, len(data))))))
    rows = math.ceil(max(1, len(data)) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    for idx, row in data.iterrows():
        col = idx % cols
        grid_row = idx // cols
        x = col * tile_w
        y = grid_row * tile_h
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 210, 210))

        image_path = row.get("preview_path", "")
        image_box = (x + 20, y + 12, x + 20 + image_size, y + 12 + image_size)
        if image_path and Path(image_path).exists():
            try:
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((image_size, image_size))
                px = x + 20 + (image_size - img.width) // 2
                py = y + 12 + (image_size - img.height) // 2
                sheet.paste(img, (px, py))
            except Exception:
                draw.rectangle(image_box, outline=(180, 80, 80))
                draw.text((x + 24, y + 76), "image error", font=font, fill=(160, 40, 40))
        else:
            draw.rectangle(image_box, outline=(180, 80, 80))
            draw.text((x + 24, y + 76), "missing image", font=font, fill=(160, 40, 40))

        text_y = y + 18 + image_size
        draw.text((x + 12, text_y), str(row.get("model_id", "")), font=font_bold, fill=(20, 20, 20))
        draw_wrapped(draw, (x + 12, text_y + 17), short_text(row.get("model_name", ""), 34), font, (30, 30, 30), 28, 14, 2)
        draw.text((x + 12, text_y + 47), f"animal: {row.get('animal_pred', '')}", font=font, fill=(20, 70, 120))
        tags = short_text(row.get("deformation_tags_pred", ""), 34)
        draw_wrapped(draw, (x + 12, text_y + 64), f"deform: {tags}", font, (120, 60, 20), 30, 14, 2)

    sheet.save(out_path)
    print(f"Wrote contact sheet to {out_path}")


if __name__ == "__main__":
    main()
