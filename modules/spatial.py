"""Spatial analysis interfaces and skeleton. Owners: Ayumi & Nic."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

try:
    from modules.classifier import Detection  # real class, used inside your repo
except ImportError:
    # Fallback for standalone testing/demo outside the full repo.
    # Same shape as assumed below: .label, .confidence, .box
    from dataclasses import dataclass

    @dataclass
    class Detection:  # type: ignore
        label: str
        confidence: float
        box: tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# ASSUMPTIONS ABOUT Detection — adjust these if your actual class differs.
# This code assumes `Detection` has:
#   .label       -> str, class name (e.g. "baby", "battery", "coin", ...)
#   .confidence  -> float
#   .box         -> tuple/list (x1, y1, x2, y2) in pixel coords, absolute
#
# If your Detection uses different attribute names (e.g. .bbox, .xywh,
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


def _label(det: Detection) -> str:
    return det.label.lower().replace(" ", "_")


def _box_xyxy(det: Detection) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = det.box
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
        child_box = _box_xyxy(child)
        child_height = _box_height(child_box)

        best_score = 0.0
        best_reason = "No hazardous objects within risk range."

        for hz in hazards:
            hz_box = _box_xyxy(hz)
            hz_label = _label(hz)

            dist_px = _edge_distance(child_box, hz_box)
            norm_dist = dist_px / child_height

            weight = DANGER_WEIGHTS.get(hz_label, DEFAULT_HAZARD_WEIGHT)
            proximity = max(0.0, 1.0 - (norm_dist / MAX_NORM_DIST))
            score = proximity * weight

            if score > best_score:
                zone = _zone(norm_dist)
                best_score = score
                best_reason = (
                    f"{hz.label} detected at {zone} proximity to child "
                    f"(~{norm_dist:.2f}x child height away, "
                    f"hazard confidence {hz.confidence:.2f})."
                )

        return round(best_score, 3), best_reason