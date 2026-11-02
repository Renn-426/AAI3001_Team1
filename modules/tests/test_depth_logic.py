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
    # Scissors overlap the person and are 0.11 nearer the camera: held in the
    # hand, within ARM_REACH_DEPTH, so depth adds no distance.
    assert result.score == .9
    assert result.level == DangerLevel.HIGH
    assert "depth diff 0.00" in result.reason
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


def _two_people_with_knife(near_conf, far_conf):
    """Person A on the left, person B on the right holding a knife."""
    from modules.classifier import Detection
    return [
        Detection("person", far_conf, (0, 0, 100, 300)),
        Detection("person", near_conf, (500, 0, 600, 300)),
        Detection("knife", 0.8, (560, 100, 580, 140)),
    ]


def test_hazard_next_to_less_confident_child_is_critical():
    """Every child is checked, not only the most confidently detected one."""
    from modules.spatial import SpatialAnalyzer

    score, reason = SpatialAnalyzer().assess(_two_people_with_knife(near_conf=0.6, far_conf=0.9), (480, 640))
    assert score == 1.0
    assert "CRITICAL" in reason


def test_child_confidence_order_does_not_change_score():
    from modules.spatial import SpatialAnalyzer

    analyzer = SpatialAnalyzer()
    low_near, _ = analyzer.assess(_two_people_with_knife(near_conf=0.6, far_conf=0.9), (480, 640))
    high_near, _ = analyzer.assess(_two_people_with_knife(near_conf=0.9, far_conf=0.6), (480, 640))
    assert low_near == high_near


def test_extra_person_far_away_keeps_single_child_score():
    from modules.classifier import Detection
    from modules.spatial import SpatialAnalyzer

    analyzer = SpatialAnalyzer()
    child = Detection("person", 0.6, (100, 100, 220, 400), depth=0.9)
    hazard = Detection("coin_battery", 0.85, (230, 150, 260, 180), depth=0.1)
    # Far in pixels and at the child's depth, so it is further from the hazard.
    far_person = Detection("person", 0.95, (560, 50, 630, 300), depth=0.9)
    alone = analyzer.assess([child, hazard], (480, 640))
    with_other = analyzer.assess([far_person, child, hazard], (480, 640))
    assert with_other == alone


def test_hazard_held_in_front_stays_high_but_hazard_behind_drops():
    from modules.spatial import SpatialAnalyzer, Detection
    person = Detection("person", .9, (200, 60, 440, 480), .6)
    held = Detection("scissors", .7, (300, 250, 360, 330), .85)     # in the hand, nearer the camera
    behind = Detection("scissors", .7, (300, 250, 360, 330), .1)    # same pixels, far behind the child
    a = SpatialAnalyzer()
    assert a.assess([person, held], (480, 640))[0] == .9
    assert a.assess([person, behind], (480, 640))[0] < .5
