"""Pluggable same-view image matching backends.

The default backend is a local, deterministic OpenCV Farneback flow. Learned
backends can implement the same interface without changing observation schema.
No backend in this module receives target xyz or a synthetic delta.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchField:
    flow: np.ndarray
    backward_flow: np.ndarray
    cycle_error: np.ndarray
    confidence: np.ndarray
    backend: str
    metadata: dict


class ImageMatcherBackend:
    name = "abstract"

    def match(self, source_rgb, target_rgb):
        raise NotImplementedError


def _gray(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.uint8, copy=False)


def _sample_flow(flow, xy):
    h, w = flow.shape[:2]
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return flow[y, x]


class FarnebackMatcher(ImageMatcherBackend):
    name = "opencv_farneback"

    def __init__(self, winsize=31, levels=4, iterations=5, poly_n=7, poly_sigma=1.5):
        self.params = dict(winsize=winsize, levels=levels, iterations=iterations,
                           poly_n=poly_n, poly_sigma=poly_sigma)

    def match(self, source_rgb, target_rgb):
        import cv2
        source = _gray(source_rgb)
        target = _gray(target_rgb)
        forward = cv2.calcOpticalFlowFarneback(
            source, target, None, 0.5, self.params["levels"],
            self.params["winsize"], self.params["iterations"],
            self.params["poly_n"], self.params["poly_sigma"], 0,
        )
        backward = cv2.calcOpticalFlowFarneback(
            target, source, None, 0.5, self.params["levels"],
            self.params["winsize"], self.params["iterations"],
            self.params["poly_n"], self.params["poly_sigma"], 0,
        )
        h, w = source.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        back_at_target = cv2.remap(backward, xx + forward[..., 0], yy + forward[..., 1],
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cycle = np.linalg.norm(forward + back_at_target, axis=-1)
        confidence = np.exp(-cycle / 2.0).astype(np.float32)
        return MatchField(forward.astype(np.float32), backward.astype(np.float32),
                          cycle.astype(np.float32), confidence, self.name,
                          {"opencv_version": cv2.__version__, **self.params})


class DISMatcher(ImageMatcherBackend):
    name = "opencv_dis"

    def __init__(self, preset="medium"):
        self.preset = preset

    def match(self, source_rgb, target_rgb):
        import cv2
        source = _gray(source_rgb)
        target = _gray(target_rgb)
        presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                   "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                   "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        flow_engine = cv2.DISOpticalFlow_create(presets.get(self.preset, presets["medium"]))
        forward = flow_engine.calc(source, target, None)
        backward = flow_engine.calc(target, source, None)
        h, w = source.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        back_at_target = cv2.remap(backward, xx + forward[..., 0], yy + forward[..., 1],
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cycle = np.linalg.norm(forward + back_at_target, axis=-1)
        confidence = np.exp(-cycle / 2.0).astype(np.float32)
        return MatchField(forward.astype(np.float32), backward.astype(np.float32),
                          cycle.astype(np.float32), confidence, self.name,
                          {"opencv_version": cv2.__version__, "preset": self.preset})


def make_matcher(name):
    if name in ("farneback", "opencv_farneback"):
        return FarnebackMatcher()
    if name in ("dis", "opencv_dis"):
        return DISMatcher()
    if name in ("learned", "raft", "dinov2"):
        return LearnedMatcherBackend()
    raise ValueError("unknown matcher backend: {}".format(name))


class LearnedMatcherBackend(ImageMatcherBackend):
    """Interface placeholder for local DINO/RAFT/LoFTR adapters.

    It intentionally fails loudly instead of downloading weights or silently
    substituting oracle projections.
    """
    name = "learned_unavailable"

    def match(self, source_rgb, target_rgb):
        raise RuntimeError("No local learned matcher weights are configured; use Farneback baseline")
