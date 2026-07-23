#!/usr/bin/env python3
"""Machine-readable preflight QC for the five-sample real target pilot.

This tool does not generate images or mine deltas.  Missing samples are
reported as pending, and observed nuisance fields remain post-hoc metadata.
"""
import argparse
import json
from pathlib import Path

from PIL import Image


EXPECTED = [f"{i:02d}" for i in range(1, 9)]


def find_image(root, prefix):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = root / f"{prefix}_standard{ext}"
        if p.exists():
            return p
        p = root / f"{prefix}{ext}"
        if p.exists():
            return p
    return None


def inspect_sample(sample_dir, source_sizes):
    images = []
    missing = []
    bad_size = []
    for prefix in EXPECTED:
        path = find_image(sample_dir, prefix)
        if path is None:
            missing.append(prefix)
            continue
        try:
            with Image.open(path) as im:
                size = list(im.size)
                mode = im.mode
            images.append({"view": prefix, "path": str(path), "size": size, "mode": mode})
            if source_sizes and tuple(size) not in source_sizes:
                bad_size.append({"view": prefix, "size": size})
        except Exception as exc:
            bad_size.append({"view": prefix, "error": str(exc)})
    return {"images": images, "missing_views": missing, "size_errors": bad_size,
            "complete": not missing and not bad_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/real_pilot")
    ap.add_argument("--source_image_root", default="assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/key8")
    ap.add_argument("--output_path", default=None)
    args = ap.parse_args()
    root = Path(args.pilot_root)
    source_root = Path(args.source_image_root)
    source_sizes = set()
    for p in source_root.glob("*.*"):
        try:
            with Image.open(p) as im:
                source_sizes.add(im.size)
        except Exception:
            pass
    samples = {}
    for name in ("sample_A", "sample_B", "sample_C", "sample_D", "sample_E"):
        sample = root / name
        samples[name] = inspect_sample(sample, source_sizes) if sample.exists() else {
            "images": [], "missing_views": EXPECTED, "size_errors": [], "complete": False,
            "status": "missing_directory"}
    manifest = root / "style_task_manifest.json"
    manifest_info = {"exists": manifest.exists()}
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            manifest_info["sample_count"] = len(data.get("samples", []))
            manifest_info["standard_prompt_shared"] = len({x.get("generation_prompt", "") for x in data.get("samples", [])}) <= 1
            manifest_info["observed_nuisance_empty"] = all(
                not x.get("observed_posthoc_nuisance", {}) for x in data.get("samples", []))
        except Exception as exc:
            manifest_info["error"] = str(exc)
    complete = all(v["complete"] for v in samples.values())
    report = {"status": "PASS" if complete else "PENDING",
              "pilot_root": str(root), "source_sizes": sorted(map(list, source_sizes)),
              "samples": samples, "manifest": manifest_info,
              "qc_rules": ["eight views per sample", "same camera filename mapping",
                            "post-hoc nuisance fields only", "manual semantic/topology review required"]}
    output = Path(args.output_path) if args.output_path else root / "pilot_qc_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "output": str(output),
                      "complete_samples": sum(v["complete"] for v in samples.values())}, indent=2))


if __name__ == "__main__":
    main()
