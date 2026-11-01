"""Spatial analysis interfaces and skeleton. Owners: Ayumi & Nic."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

try:
    from modules.classifier import Detection  # real class, used inside your repo
except ImportError:
    # Fallback for standalone testing/demo outside the full repo.
    # Same shape as the real class: .label, .confidence, .bbox, .depth
    from dataclasses import dataclass

    @dataclass
    class Detection:  # type: ignore
        label: str
        confidence: float
        bbox: tuple[float, float, float, float] | None = None
        depth: float | None = None  # optional: 0-1 normalized relative depth
        # (1 = closest to camera, 0 = farthest). Leave as None if you're
        # not using depth estimation - everything still works, just
        # falls back to planar-only distance.

# ---------------------------------------------------------------------------
# ASSUMPTIONS ABOUT Detection — adjust these if your actual class differs.
# This code assumes `Detection` has:
#   .label       -> str, class name (e.g. "baby", "battery", "coin", ...)
#   .confidence  -> float
#   .bbox        -> tuple/list (x1, y1, x2, y2) in pixel coords, absolute
#                   (detections with bbox=None are ignored)
#   .depth       -> optional float, 0-1 relative depth (1 = closest)
#
# If your Detection uses different attribute names (e.g. .xywh,
# .cls_name) or a normalized box, change the `_box_xyxy` and `_label`
# helper functions below — nothing else needs to change.
# ---------------------------------------------------------------------------

# Class name(s) your child detector uses. Default assumes a COCO-pretrained
# YOLO model (yolov8n.pt / yolov8l.pt) used for its "person" class — the
# most reliable off-the-shelf option (see conversation). If you swap to a
# baby-specific model instead (e.g. kairess/baby-face-detection-yolov10),
# change this to {"baby"}.
CHILD_LABELS = {"person"}

# Hazard class names -> danger weight (0-1), matched to the 14-class
# hazard label list (class IDs 0-13):
#   0 battery          7 pen
#   1 coin_battery      8 pencap
#   2 bottlecap         9 knife
#   3 coin              10 scissors
#   4 dice              11 liquid_medicine
#   5 marble            12 pill_bottle
#   6 small_toy         13 pill_packet
# Weights are a starting point based on choking/poisoning/laceration
# severity — adjust to your own judgment or a cited safety reference.
DANGER_WEIGHTS = {
    "battery": 0.85,          # standard AA/AAA - choking + chemical risk if punctured
    "coin_battery": 1.0,      # button batteries - most severe, can burn esophagus within hours
    "bottlecap": 0.6,
    "coin": 0.8,
    "dice": 0.6,
    "marble": 0.85,           # classic choking-hazard size
    "small_toy": 0.75,
    "pen": 0.3,                # low risk itself; mostly a precursor to pencap
    "pencap": 0.65,
    "knife": 1.0,              # laceration risk, highest severity
    "scissors": 0.9,
    "liquid_medicine": 0.95,   # poisoning risk, especially if flavored/syrup
    "pill_bottle": 0.9,        # risk depends on whether it's open/childproofed
    "pill_packet": 0.95,       # blister packs are easy for kids to open, high poisoning risk
}
DEFAULT_HAZARD_WEIGHT = 0.5

# Normalized-distance zone thresholds (fraction of child box height).
ZONE_THRESHOLDS = {
    "CRITICAL": 0.5,
    "WARNING": 1.5,
    "CAUTION": 3.0,
}
MAX_NORM_DIST = ZONE_THRESHOLDS["CAUTION"]  # beyond this, risk score is 0

# How much a depth mismatch (two boxes close in pixels but at very
# different distances from the camera) inflates the effective distance.
# depth is expected 0-1 normalized (1 = closest to camera). A depth_diff
# of e.g. 0.3 gets multiplied by this weight and combined with the planar
# distance via sqrt(planar^2 + (weight*depth_diff)^2) - i.e. it behaves
# as an extra "distance" axis, same units as norm_dist. Tune by watching
# whether foreground/background false positives go away on your footage.
DEPTH_DIFF_WEIGHT = 3.0


def _label(det: Detection) -> str:
    return det.label.lower().replace(" ", "_")


def _box_xyxy(det: Detection) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = det.bbox
    return float(x1), float(y1), float(x2), float(y2)


def _box_height(box: tuple[float, float, float, float]) -> float:
    _, y1, _, y2 = box
    return max(y2 - y1, 1e-6)


def _edge_distance(box_a, box_b) -> float:
    """Shortest distance between two boxes' edges; 0 if they overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.sqrt(dx * dx + dy * dy)


def _zone(norm_dist: float) -> str:
    if norm_dist < ZONE_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if norm_dist < ZONE_THRESHOLDS["WARNING"]:
        return "WARNING"
    if norm_dist < ZONE_THRESHOLDS["CAUTION"]:
        return "CAUTION"
    return "SAFE"


def depth_adjusted_distance(child: Detection, hazard: Detection) -> tuple[float, float | None, float]:
    """(planar distance, depth difference, effective distance) from child
    to hazard, in child box heights. Depth difference is None and effective
    equals planar unless both detections have a depth value."""
    child_box = _box_xyxy(child)
    norm_dist = _edge_distance(child_box, _box_xyxy(hazard)) / _box_height(child_box)

    child_depth = getattr(child, "depth", None)
    hz_depth = getattr(hazard, "depth", None)
    if child_depth is None or hz_depth is None:
        return norm_dist, None, norm_dist  # fall back to planar-only

    depth_diff = abs(child_depth - hz_depth)
    effective_dist = math.sqrt(norm_dist ** 2 + (DEPTH_DIFF_WEIGHT * depth_diff) ** 2)
    return norm_dist, depth_diff, effective_dist


class ScoreHold:
    """Keeps the highest recent score for `seconds`, so one missed detection
    doesn't drop a real danger to 0. Assessment itself stays per-frame."""

    def __init__(self, seconds: float = 1.0) -> None:
        self.seconds = seconds
        self._score = 0.0
        self._reason = ""
        self._time = float("-inf")

    def update(self, score: float, reason: str, now: float) -> tuple[float, str, bool]:
        """Return (score, reason, held); held=True if an older score won."""
        if score >= self._score or now - self._time > self.seconds:
            self._score, self._reason, self._time = score, reason, now
            return score, reason, False
        return self._score, self._reason, True


class BaseSpatialAnalyzer(ABC):
    @abstractmethod
    def assess(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> tuple[float, str]:
        """Return (danger_score, reason) for detections in a frame.

        frame_shape is (height, width). Score must be in [0, 1]; reason
        explains the spatial risk for the UI and Telegram alerts.
        """


class SpatialAnalyzer(BaseSpatialAnalyzer):
    """Spatial teammates implement proximity and risk assessment here."""

    def assess(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> tuple[float, str]:
        detections = [d for d in detections if d.bbox is not None]
        children = [d for d in detections if _label(d) in CHILD_LABELS]
        hazards = [d for d in detections if _label(d) not in CHILD_LABELS
                   and _label(d) in DANGER_WEIGHTS]

        # --- no child in frame: nothing to assess against ---
        if not children:
            return 0.0, "No child detected in frame."

        # --- no hazards in frame: child is safe regardless of position ---
        if not hazards:
            return 0.0, "No hazardous objects detected near the child."

        # Use the highest-confidence child box as the reference point.
        # (If you expect multiple children, loop over all of them and
        # take the max risk across children instead.)
        child = max(children, key=lambda d: d.confidence)

        best_score = 0.0
        best_reason = "No hazardous objects within risk range."

        for hz in hazards:
            hz_label = _label(hz)

            # --- depth-aware adjustment (only if both have depth values) ---
            _, depth_diff, effective_dist = depth_adjusted_distance(child, hz)
            depth_note = "" if depth_diff is None else f", depth diff {depth_diff:.2f}"

            weight = DANGER_WEIGHTS.get(hz_label, DEFAULT_HAZARD_WEIGHT)
            proximity = max(0.0, 1.0 - (effective_dist / MAX_NORM_DIST))
            score = proximity * weight

            if score > best_score:
                zone = _zone(effective_dist)
                best_score = score
                best_reason = (
                    f"{hz.label} detected at {zone} proximity to child "
                    f"(~{effective_dist:.2f}x child height away{depth_note}, "
                    f"hazard confidence {hz.confidence:.2f})."
                )

        return round(best_score, 3), best_reason