#!/usr/bin/env python3
"""Score weak generated standard references and prepare Tripo-ready input."""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
from PIL import Image


DEFAULT_REPO_ROOT = "/home/shichang/Deformable-3D-Gaussians"
DEFAULT_OBJECT_ID = "big_carved_wooden_elephant_sculpture"
DEFAULT_PREPARED = os.path.join(DEFAULT_REPO_ROOT, "assets/prepared", DEFAULT_OBJECT_ID)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views_meta", default=os.path.join(DEFAULT_PREPARED, "renders_original/key8/views_meta.json"))
    parser.add_argument("--generated_dir", default=os.path.join(DEFAULT_PREPARED, "generated_standard/key8_raw"))
    parser.add_argument("--selected_dir", default=os.path.join(DEFAULT_PREPARED, "generated_standard/selected"))
    parser.add_argument("--tripo_dir", default=os.path.join(DEFAULT_PREPARED, "tripo_input"))
    parser.add_argument("--report", default=os.path.join(DEFAULT_PREPARED, "generated_standard/selection_report.json"))
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--min_front_score", type=float, default=0.35)
    parser.add_argument("--use_clip", action="store_true")
    return parser.parse_args()


def load_image(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def foreground_mask(rgba):
    alpha = rgba[..., 3]
    if alpha.max() < 0.999 or alpha.min() > 0.001:
        return alpha > 0.2
    rgb = rgba[..., :3]
    near_white = np.all(rgb > 0.96, axis=-1)
    near_black = np.all(rgb < 0.03, axis=-1)
    return ~(near_white | near_black)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def contour_similarity(a, b):
    box_a = bbox(a)
    box_b = bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    h, w = a.shape
    norm = np.array([w, h, w, h], dtype=np.float32)
    box_a = box_a / norm
    box_b = box_b / norm
    box_score = 1.0 - np.clip(np.abs(box_a - box_b).mean() * 4.0, 0.0, 1.0)
    area_a = max(float(a.mean()), 1e-6)
    area_b = max(float(b.mean()), 1e-6)
    area_score = 1.0 - min(abs(area_a - area_b) / max(area_a, area_b), 1.0)
    cy_a, cx_a = np.argwhere(a).mean(axis=0)
    cy_b, cx_b = np.argwhere(b).mean(axis=0)
    center_dist = np.sqrt(((cx_a - cx_b) / w) ** 2 + ((cy_a - cy_b) / h) ** 2)
    center_score = 1.0 - min(center_dist * 4.0, 1.0)
    return float(0.45 * box_score + 0.35 * area_score + 0.20 * center_score)


def clip_similarity_unavailable():
    return None, "CLIP scoring skipped. Use --use_clip with a locally installed/configured CLIP stack if needed."


def maybe_clip_similarity(original_path, generated_path, use_clip):
    if not use_clip:
        return clip_similarity_unavailable()
    try:
        import torch
        import open_clip
    except Exception as exc:
        return None, "CLIP scoring unavailable: {}".format(exc)
    try:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        img_a = preprocess(Image.open(original_path).convert("RGB")).unsqueeze(0).to(device)
        img_b = preprocess(Image.open(generated_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_a = model.encode_image(img_a)
            feat_b = model.encode_image(img_b)
            feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            return float((feat_a * feat_b).sum().item()), None
    except Exception as exc:
        return None, "CLIP scoring failed: {}".format(exc)


def generated_path_for(generated_dir, source_filename):
    stem = os.path.splitext(os.path.basename(source_filename))[0]
    return os.path.join(generated_dir, stem + "_standard.png")


def copy_selected(scored, selected_dir):
    os.makedirs(selected_dir, exist_ok=True)
    outputs = []
    for rank, item in enumerate(scored, 1):
        dest = os.path.join(selected_dir, "%02d_%s" % (rank, os.path.basename(item["generated_image"])))
        shutil.copy2(item["generated_image"], dest)
        item["selected_copy"] = dest
        outputs.append(dest)
    return outputs


def choose_tripo(scored, min_front_score):
    front = [x for x in scored if "front three-quarter" in x.get("view_hint", "")]
    front = sorted(front, key=lambda x: x["total_score"], reverse=True)
    if front and front[0]["total_score"] >= min_front_score:
        return front[0], "preferred front three-quarter view"
    return max(scored, key=lambda x: x["total_score"]), "front three-quarter score too low or unavailable; using best overall"


def main():
    args = parse_args()
    with open(args.views_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scored = []
    warnings = []
    base_dir = os.path.dirname(os.path.abspath(args.views_meta))

    for view in meta["views"]:
        original = view.get("path") or os.path.join(base_dir, view["filename"])
        generated = generated_path_for(args.generated_dir, view["filename"])
        if not os.path.isfile(generated):
            warnings.append("Missing generated image for {}".format(view["filename"]))
            continue

        orig_rgba = load_image(original)
        gen_rgba = load_image(generated)
        orig_mask = foreground_mask(orig_rgba)
        gen_mask = foreground_mask(gen_rgba)
        iou = mask_iou(orig_mask, gen_mask)
        contour = contour_similarity(orig_mask, gen_mask)
        clip_sim, clip_warning = maybe_clip_similarity(original, generated, args.use_clip)
        if clip_warning and clip_warning not in warnings:
            warnings.append(clip_warning)

        total = 0.70 * iou + 0.30 * contour
        if clip_sim is not None:
            total = 0.55 * iou + 0.25 * contour + 0.20 * max(clip_sim, 0.0)

        scored.append(
            {
                "filename": view["filename"],
                "view_name": view.get("name"),
                "view_hint": view.get("hint", ""),
                "source_image": original,
                "generated_image": generated,
                "scores": {
                    "mask_iou": iou,
                    "contour_similarity": contour,
                    "clip_similarity": clip_sim,
                },
                "total_score": float(total),
            }
        )

    scored = sorted(scored, key=lambda x: x["total_score"], reverse=True)
    selected = scored[: args.keep]
    selected_paths = copy_selected(selected, args.selected_dir)

    os.makedirs(args.tripo_dir, exist_ok=True)
    chosen = None
    chosen_reason = None
    if scored:
        chosen, chosen_reason = choose_tripo(scored, args.min_front_score)
        tripo_path = os.path.join(args.tripo_dir, "standard_front_3quarter.png")
        shutil.copy2(chosen["generated_image"], tripo_path)
        chosen["tripo_copy"] = tripo_path
        if chosen["total_score"] < 0.35:
            warnings.append("Chosen Tripo input has low similarity score: %.3f" % chosen["total_score"])
    else:
        tripo_path = None
        warnings.append("No generated images were available for selection.")

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "views_meta": os.path.abspath(args.views_meta),
        "generated_dir": os.path.abspath(args.generated_dir),
        "per_view": scored,
        "selected_images": selected_paths,
        "chosen_tripo_input": tripo_path,
        "chosen_tripo_reason": chosen_reason,
        "warnings": warnings,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote selection report:", args.report)
    if tripo_path:
        print("Tripo-ready image:", tripo_path)


if __name__ == "__main__":
    main()
