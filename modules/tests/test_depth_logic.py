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


def test_ui_adapter_preserves_demo_depth_and_score():
    """The UI bridge must reproduce the demo's 0.11 depth-difference example."""
    import threading
    from types import SimpleNamespace
    import numpy as np
    from modules.classifier import SpatialDemoClassifier, DangerLevel
    from modules.spatial import SpatialAnalyzer

    def box(coords, class_id, confidence):
        return SimpleNamespace(xyxy=np.array([coords]), cls=np.array([class_id]),
                               conf=np.array([confidence]))

    prediction = SimpleNamespace(
        names={0: "person", 76: "scissors"},
        boxes=[box((100, 100, 300, 400), 0, .9), box((200, 200, 260, 260), 76, .85)],
    )
    bridge = object.__new__(SpatialDemoClassifier)
    bridge.model = SimpleNamespace(predict=lambda *args, **kwargs: [prediction])
    bridge.wanted = [0, 76]
    bridge._lock = threading.Lock()
    bridge.spatial = SpatialAnalyzer()
    bridge.depth_error = None

    depth = np.zeros((480, 640), dtype=float)
    depth[:, 320:] = 1.0
    depth[175:325, 150:250] = .03
    depth[215:245, 215:245] = .14
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = bridge.predict_with_depth(frame, depth)
    assert result.score == .801
    assert result.level == DangerLevel.HIGH
    assert "~0.33x" in result.reason and "depth diff 0.11" in result.reason
    assert result.depth_status == "Relative depth active"

    planar = bridge.predict_with_depth(frame, None)
    assert planar.score == .9
    assert "depth diff" not in planar.reason
    assert "Image-plane distance only" in planar.depth_status

    bridge.depth_error = "test depth failure"
    failed_depth = bridge.predict_with_depth(frame, None)
    assert failed_depth.score == .9
    assert "Depth unavailable" in failed_depth.depth_status
    assert "test depth failure" in failed_depth.depth_status
