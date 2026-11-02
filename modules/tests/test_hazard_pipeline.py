"""Trained hazard model + person model + spatial analysis, with stand-in models.

    python -m pytest modules/tests/test_hazard_pipeline.py -q
"""
import threading
from types import SimpleNamespace

import numpy as np

from modules.classifier import DangerLevel, HazardSpatialClassifier
from modules.spatial import SpatialAnalyzer


def _box(coords, class_id, confidence):
    return SimpleNamespace(xyxy=np.array([coords]), cls=np.array([class_id]),
                           conf=np.array([confidence]))


def _pipeline(hazard_boxes, hazard_names, people=True):
    person = SimpleNamespace(names={0: "person"},
                             boxes=[_box((100, 100, 300, 400), 0, .9)] if people else [])
    hazards = SimpleNamespace(names=hazard_names, boxes=hazard_boxes)
    clf = object.__new__(HazardSpatialClassifier)          # skip loading real models
    clf.model = SimpleNamespace(predict=lambda *a, **k: [person])
    clf.wanted = [0]
    clf.detector = SimpleNamespace(predict_raw=lambda frame: hazards)
    clf._lock = threading.Lock()
    clf.spatial = SpatialAnalyzer()
    clf.depth_error = None
    return clf


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_trained_hazard_next_to_person_is_high():
    clf = _pipeline([_box((280, 200, 320, 240), 10, .8)], {10: "coin_battery"})
    r = clf.predict_with_depth(FRAME, None)
    assert {d.label for d in r.detections} == {"person", "coin_battery"}
    assert r.level == DangerLevel.HIGH and "coin_battery" in r.reason


def test_far_hazard_scores_lower_than_near_hazard():
    near = _pipeline([_box((280, 200, 320, 240), 1, .8)], {1: "knife"}).predict_with_depth(FRAME, None)
    far = _pipeline([_box((600, 440, 630, 470), 1, .8)], {1: "knife"}).predict_with_depth(FRAME, None)
    assert near.score > far.score


def test_hazard_without_person_is_safe():
    r = _pipeline([_box((280, 200, 320, 240), 1, .9)], {1: "knife"}, people=False).predict_with_depth(FRAME, None)
    assert r.level == DangerLevel.SAFE and "No child" in r.reason


def test_depth_difference_lowers_the_score():
    clf = _pipeline([_box((200, 200, 260, 260), 1, .85)], {1: "scissors"})
    depth = np.zeros((480, 640), float)
    depth[:, 320:] = 1.0
    depth[175:325, 150:250] = .03                            # person far away
    depth[215:245, 215:245] = .9                             # hazard close to the camera
    assert clf.predict_with_depth(FRAME, depth).score < clf.predict_with_depth(FRAME, None).score
