#!/usr/bin/env python3
"""Create per-view prompts for standardizing a stylized elephant sculpture render."""

import argparse
import json
import os
import shutil
from datetime import datetime


DEFAULT_REPO_ROOT = "/home/shichang/Deformable-3D-Gaussians"
DEFAULT_OBJECT_ID = "big_carved_wooden_elephant_sculpture"
DEFAULT_PREPARED = os.path.join(DEFAULT_REPO_ROOT, "assets/prepared", DEFAULT_OBJECT_ID)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views_meta", default=os.path.join(DEFAULT_PREPARED, "renders_original/key8/views_meta.json"))
    parser.add_argument("--output", default=os.path.join(DEFAULT_PREPARED, "prompts/prompts_standard_key8.json"))
    parser.add_argument("--manual_md", default=os.path.join(DEFAULT_PREPARED, "prompts/chatgpt_manual_prompts.md"))
    parser.add_argument("--upload_pack", default=os.path.join(DEFAULT_PREPARED, "chatgpt_upload_pack"))
    parser.add_argument("--object_category", default="elephant sculpture")
    return parser.parse_args()


def view_hint(view):
    hint = view.get("hint")
    if hint:
        return hint
    name = view.get("filename", "").lower()
    if "front_3quarter" in name or "az045" in name or "az315" in name:
        return "front three-quarter view"
    if "front" in name or "az000" in name:
        return "front view"
    if "side" in name or "az090" in name or "az270" in name:
        return "side view"
    if "back_3quarter" in name or "az135" in name or "az225" in name:
        return "back three-quarter view"
    if "back" in name or "az180" in name:
        return "back view"
    return "same camera viewpoint"


def build_prompt(category, hint):
    return """Convert this rendered stylized {category} into a less stylized, more standard {category}.

View-specific hint: {hint}.

Strict constraints:
- Preserve exactly the same camera viewpoint and object orientation.
- Preserve the same object category: {category}.
- Preserve the same number of major parts and their spatial layout.
- Preserve all visible major parts, appendages, support/base elements, and their spatial layout.
- Preserve the overall silhouette as much as possible.
- Do not add, remove, or move major parts.
- Keep the object centered and isolated on a plain white or transparent background.
- Do not add text, labels, people, plants, extra props, or a scene background.

Style reduction:
- Reduce overly carved, ornamental, exaggerated, or stylized details.
- Reduce cartoon-like or toy-like proportions if present.
- Make the geometry more standard, clean, and realistic.
- Keep it as a sculpture/object, not a living subject.
- Keep material simple and not overly decorative.

Output:
- A clean single-object render.
- Same view as the input image.
- Plain background.
- No extra objects.""".format(category=category, hint=hint)


def pack_filename(rank, view):
    elev = int(round(float(view.get("elevation", 0.0))))
    az = int(round(float(view.get("azimuth", 0.0)))) % 360
    return "{:02d}_key_e{:03d}_a{:03d}.png".format(rank, elev, az)


def write_manual_markdown(prompts, path):
    lines = [
        "# Manual ChatGPT Prompts",
        "",
        "Upload each image from `chatgpt_upload_pack/` to ChatGPT with the matching prompt text.",
        "Download the generated result using the same numeric prefix, for example `01_standard.png`.",
        "",
    ]
    for item in prompts["prompts"]:
        lines.extend(
            [
                "## {} / {}".format(item["pack_image"], item["view_hint"]),
                "",
                "Image: `{}`".format(item["pack_image"]),
                "",
                "Prompt:",
                "",
                "```text",
                item["prompt"],
                "```",
                "",
            ]
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_upload_pack(prompts, path):
    os.makedirs(path, exist_ok=True)
    for item in prompts["prompts"]:
        image_dest = os.path.join(path, item["pack_image"])
        prompt_dest = os.path.join(path, item["pack_prompt"])
        shutil.copy2(item["source_image"], image_dest)
        with open(prompt_dest, "w", encoding="utf-8") as f:
            f.write(item["prompt"].rstrip() + "\n")


def main():
    args = parse_args()
    with open(args.views_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    prompts = {
        "object_id": meta.get("object_id", DEFAULT_OBJECT_ID),
        "source_views_meta": os.path.abspath(args.views_meta),
        "object_category": args.object_category,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "prompts": [],
    }

    base_dir = os.path.dirname(os.path.abspath(args.views_meta))
    for rank, view in enumerate(meta["views"], 1):
        hint = view_hint(view)
        source_image = view.get("path") or os.path.join(base_dir, view["filename"])
        image_name = pack_filename(rank, view)
        prompts["prompts"].append(
            {
                "view_index": view.get("index"),
                "view_name": view.get("name"),
                "filename": view["filename"],
                "source_image": source_image,
                "view_hint": hint,
                "azimuth": view.get("azimuth"),
                "elevation": view.get("elevation"),
                "pack_image": image_name,
                "pack_prompt": "{:02d}_prompt.txt".format(rank),
                "prompt": build_prompt(args.object_category, hint),
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)
    print("Wrote prompts:", args.output)
    write_manual_markdown(prompts, args.manual_md)
    print("Wrote manual prompt markdown:", args.manual_md)
    write_upload_pack(prompts, args.upload_pack)
    print("Wrote ChatGPT upload pack:", args.upload_pack)


if __name__ == "__main__":
    main()
