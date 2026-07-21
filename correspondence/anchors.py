"""Semantic anchor schema helpers."""
import json


def load_anchors(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    anchors = payload.get("anchors", payload)
    required = {"name", "source_xyz", "target_xyz", "confidence"}
    for anchor in anchors:
        missing = required.difference(anchor)
        if missing: raise ValueError("anchor missing fields: {}".format(sorted(missing)))
    return anchors


def paired_anchor_points(anchors, min_confidence=0.0):
    pairs = [a for a in anchors if a["source_xyz"] is not None and a["target_xyz"] is not None and float(a.get("confidence", 0)) >= min_confidence]
    if len(pairs) < 3: raise ValueError("at least three confident paired anchors are required")
    return pairs
