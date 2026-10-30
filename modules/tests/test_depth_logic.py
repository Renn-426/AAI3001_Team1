"""
Quick test of the depth-aware distance logic in spatial_analyzer.py,
WITHOUT running an actual depth model - just checking that providing
depth values changes the outcome the way we expect.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial import SpatialAnalyzer, Detection

analyzer = SpatialAnalyzer()
baby_box = (100, 100, 220, 400)
hazard_box_overlap = (230, 150, 260, 180)  # close in pixels to baby box

print("=== Case A: same depth (both close to camera) - should be CRITICAL ===")
dets = [
    Detection(label="person", confidence=0.9, bbox=baby_box, depth=0.9),
    Detection(label="coin_battery", confidence=0.85, bbox=hazard_box_overlap, depth=0.88),
]
score, reason = analyzer.assess(dets, (480, 640))
print(f"  score={score}  reason={reason}\n")

print("=== Case B: very different depth (hazard far in background) - should drop ===")
dets = [
    Detection(label="person", confidence=0.9, bbox=baby_box, depth=0.9),
    Detection(label="coin_battery", confidence=0.85, bbox=hazard_box_overlap, depth=0.1),
]
score, reason = analyzer.assess(dets, (480, 640))
print(f"  score={score}  reason={reason}\n")

print("=== Case C: no depth info provided (planar-only fallback) ===")
dets = [
    Detection(label="person", confidence=0.9, bbox=baby_box),
    Detection(label="coin_battery", confidence=0.85, bbox=hazard_box_overlap),
]
score, reason = analyzer.assess(dets, (480, 640))
print(f"  score={score}  reason={reason}")
