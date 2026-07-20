#!/usr/bin/env python3
"""Composite RGBA images over a plain background, preserving filenames."""

import argparse
import os
import shutil

import numpy as np
from PIL import Image


EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--background", default="white", choices=("white",))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(root, recursive=False):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(EXTS):
                    yield os.path.join(dirpath, filename)
    else:
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if os.path.isfile(path) and filename.lower().endswith(EXTS):
                yield path


def composite_to_white(src, dst):
    image = Image.open(src)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if not has_alpha:
        shutil.copy2(src, dst)
        return False
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8), "RGB").save(dst)
    return True


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)
    processed = 0
    alpha_count = 0
    copied = 0
    for src in iter_images(input_dir, args.recursive):
        rel = os.path.relpath(src, input_dir)
        dst = os.path.join(out_dir, rel)
        if os.path.exists(dst) and not args.overwrite:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        had_alpha = composite_to_white(src, dst)
        processed += 1
        alpha_count += int(had_alpha)
        copied += int(not had_alpha)
    print("input_dir:", input_dir)
    print("out_dir:", out_dir)
    print("images processed:", processed)
    print("images with alpha composited:", alpha_count)
    print("RGB images copied unchanged:", copied)


if __name__ == "__main__":
    main()
