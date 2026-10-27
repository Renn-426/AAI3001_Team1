"""
Standalone demo of the distance/risk math in spatial_analyzer.py.

No models, no images needed — just plug in bounding box coordinates
(x1, y1, x2, y2) and see what the analyzer computes. Useful for sanity
checking the geometry before any detector is plugged in.

Run:
    python test_distance_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial import (
    _box_xyxy, _box_height, _edge_distance, _zone,
    SpatialAnalyzer, Detection, DANGER_WEIGHTS, MAX_NORM_DIST,
)


def manual_distance_check(box_a, box_b, label_a="box_a", label_b="box_b"):
    """Directly test the geometry: distance, normalized distance, zone."""
    dist_px = _edge_distance(box_a, box_b)
    height_a = _box_height(box_a)
    norm_dist = dist_px / height_a
    zone = _zone(norm_dist)

    print(f"--- {label_a} vs {label_b} ---")
    print(f"  {label_a} box: {box_a}  (height={height_a:.0f}px)")
    print(f"  {label_b} box: {box_b}")
    print(f"  raw pixel distance:   {dist_px:.1f}px")
    print(f"  normalized distance:  {norm_dist:.2f}x  (fraction of {label_a}'s box height)")
    print(f"  zone:                 {zone}")
    print()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example 1: two boxes close together (simulating e.g. baby #1 vs
    # baby #2, or any two boxes you want to sanity-check distance on)
    # ------------------------------------------------------------------
    baby1_box = (100, 100, 220, 400)   # x1,y1,x2,y2 - e.g. 120px wide, 300px tall
    baby2_box = (250, 150, 370, 420)   # close by, slightly to the right

    manual_distance_check(baby1_box, baby2_box, "baby_1", "baby_2")

    # ------------------------------------------------------------------
    # Example 2: same two boxes, but far apart
    # ------------------------------------------------------------------
    baby2_far_box = (800, 150, 920, 420)
    manual_distance_check(baby1_box, baby2_far_box, "baby_1", "baby_2_far")

    # ------------------------------------------------------------------
    # Example 3: using the REAL SpatialAnalyzer.assess() with a baby box
    # and a hazard box, the way it's actually meant to be used
    # ------------------------------------------------------------------
    print("=== Full SpatialAnalyzer.assess() example (baby + hazard) ===")
    analyzer = SpatialAnalyzer()

    detections = [
        Detection(label="person", confidence=0.91, box=baby1_box),
        Detection(label="coin_battery", confidence=0.87, box=(230, 380, 260, 410)),
    ]
    frame_shape = (480, 640)  # (height, width)

    score, reason = analyzer.assess(detections, frame_shape)
    print(f"  score: {score}")
    print(f"  reason: {reason}")
