#!/usr/bin/env python3
"""Placeholder adapter for external Gaussian sources.

This script intentionally does not fake a conversion. Different source builders
store scaling, rotation, opacity, and color/SH fields with different schemas.
Implement a format-specific adapter once the external output format is known.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="External Gaussian file or directory.")
    parser.add_argument("--output", required=True, help="Target local GaussianModel-compatible PLY path.")
    parser.add_argument(
        "--source_format",
        choices=("graphdeco", "gsplat", "splatfacto", "unknown"),
        default="unknown",
    )
    parser.add_argument("--camera_dataset", help="Optional matching Blender/NeRF camera dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("External Gaussian conversion is not implemented yet.")
    print("Input:", os.path.abspath(args.input))
    print("Requested output:", os.path.abspath(args.output))
    print("Source format:", args.source_format)
    if args.camera_dataset:
        print("Camera dataset:", os.path.abspath(args.camera_dataset))
    print("")
    print("TODO:")
    print("- inspect the external file schema;")
    print("- map xyz, scaling, rotation, opacity, and SH/color fields;")
    print("- write this repo's GaussianModel-compatible PLY;")
    print("- verify by rendering with source_quality_gate.py.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
