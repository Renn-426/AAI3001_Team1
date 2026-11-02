"""Shared glue between YOLO, depth and SpatialAnalyzer. Owners: Ayumi & Nic.

Used by the app (classifier.SpatialDemoClassifier) and the live test script
(modules/tests/live_spatial_test.py), so both convert detections, sample
depth and apply thresholds the same way.
"""
from __future__ import annotations

import cv2
import numpy as np

from modules.classifier import Detection

# YOLO settings shared by the app and the test script's defaults.
IMGSZ = 640          # model input size; 960 finds smaller objects but is ~2x slower on CPU
HAZARD_CONF = 0.2    # minimum hazard confidence
CHILD_CONF = 0.35    # minimum child confidence (kept higher to avoid false children)

# Live depth maps older than this (seconds) are ignored and planar distance is
# used instead. Depth can only lower a score, so an old map that predates a
# hazard's arrival would hide real danger; planar distance never under-warns.
MAX_DEPTH_AGE = 1.5


def yolo_to_detections(result, depth_map, offset=(0, 0)) -> list[Detection]:
    """Convert one ultralytics Results into Detections, adding each box's
    0-1 depth when a depth map is available. offset moves boxes from a crop
    back to full-frame pixels."""
    if depth_map is not None:
        from modules.depth_estimator import depth_range, normalize_depth_value, sample_box_depth
        d_range = depth_range(depth_map)

    ox, oy = offset
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        depth = None
        if depth_map is not None:
            depth = normalize_depth_value(sample_box_depth(depth_map, box), depth_map, d_range)
        dets.append(Detection(
            label=result.names[int(b.cls[0])],
            confidence=float(b.conf[0]),
            bbox=box,
            depth=depth,
        ))
    return dets


def colorize_depth(depth_map, size=None) -> np.ndarray:
    """Depth map as a BGR image for display (brighter = closer).
    size = (width, height) to resize to, e.g. the camera frame's size."""
    gray = cv2.normalize(np.asarray(depth_map, dtype=np.float32), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    if size is not None:
        gray = cv2.resize(gray, size)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
