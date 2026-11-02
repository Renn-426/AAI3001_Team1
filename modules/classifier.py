"""Shared result types and detector -> spatial -> UI integration.

Tze owns detector.py; Ayumi & Nic own spatial.py. SafetyClassifier connects
their interfaces without changing the UI's predict() contract.
"""
from __future__ import annotations

import random
import os
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.detector import BaseDetector
    from modules.spatial import BaseSpatialAnalyzer


class DangerLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def color(self) -> str:
        return {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}[int(self)]


@dataclass
class Detection:
    """One detected object/person. bbox = (x1, y1, x2, y2) in pixels."""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    depth: float | None = None        # optional 0-1 relative depth (1 = closest)


@dataclass
class ClassificationResult:
    level: DangerLevel
    score: float                      # 0.0 - 1.0 raw danger score
    detections: list[Detection] = field(default_factory=list)
    reason: str = ""                  # human-readable, goes into the Telegram alert
    depth_status: str = ""
    score_held: bool = False


# Score -> level thresholds (tune later, or expose in the UI sidebar)
DEFAULT_THRESHOLDS = {
    DangerLevel.LOW: 0.25,
    DangerLevel.MEDIUM: 0.50,
    DangerLevel.HIGH: 0.75,
}


def score_to_level(score: float, thresholds: dict | None = None) -> DangerLevel:
    t = thresholds or DEFAULT_THRESHOLDS
    level = DangerLevel.SAFE
    for lvl in (DangerLevel.LOW, DangerLevel.MEDIUM, DangerLevel.HIGH):
        if score >= t[lvl]:
            level = lvl
    return level


class BaseClassifier(ABC):
    """Interface every classifier must follow."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        """frame_bgr: HxWx3 uint8 array (OpenCV BGR order)."""


class SafetyClassifier(BaseClassifier):
    """Connect an implemented detector and spatial analyzer to the UI."""

    def __init__(
        self, detector: BaseDetector, spatial: BaseSpatialAnalyzer
    ) -> None:
        self.detector = detector
        self.spatial = spatial

    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        detections = self.detector.detect(frame_bgr)
        score, reason = self.spatial.assess(detections, frame_bgr.shape[:2])
        if not 0.0 <= score <= 1.0:
            raise ValueError("Spatial danger score must be between 0 and 1.")
        return ClassificationResult(
            level=score_to_level(score),
            score=score,
            detections=detections,
            reason=reason,
        )


class DummyClassifier(BaseClassifier):
    """Random scores for UI development while teammates implement the pipeline."""

    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        score = random.random() ** 2          # skewed towards safe
        return ClassificationResult(
            level=score_to_level(score),
            score=score,
            detections=[],
            reason="Dummy classifier (no real model connected)",
        )


class SpatialDemoClassifier(BaseClassifier):
    """Adapt the spatial team's existing webcam demo to the UI's predict API.

    Stock COCO YOLO only: finds the person (child reference) plus the two hazards
    COCO knows (knife, scissors). Used when Tze's trained model is not available.
    """

    mode = "Stock model: person, knife and scissors only"
    imgsz = 960

    def __init__(self) -> None:
        import torch
        from ultralytics import YOLO
        from modules.spatial import CHILD_LABELS, DANGER_WEIGHTS, SpatialAnalyzer

        # Downloaded automatically on first run if missing.
        self.weights_path = str(Path(__file__).resolve().parents[1] / "weights" / "yolov8s.pt")
        self.model = YOLO(self.weights_path)
        self.names = self.model.names
        self.wanted = [i for i, name in self.names.items()
                       if name in CHILD_LABELS or name.lower().replace(" ", "_") in DANGER_WEIGHTS]
        self.spatial = SpatialAnalyzer()
        self._lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self.depth_error = None
        self.model.predict(np.zeros((480, 640, 3), np.uint8), conf=.2,
                           imgsz=self.imgsz, classes=self.wanted, verbose=False)
        # YOLO setup resets the CPU pool; limit it after setup so video has CPU time.
        torch.set_num_threads(2)

        # Reuse the downloaded depth model without waiting for network metadata.
        cache = Path(self.weights_path).parent / "huggingface"
        if cache.is_dir():
            os.environ.setdefault("HF_HOME", str(cache))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from modules.depth_estimator import compute_depth_map
            compute_depth_map(np.zeros((480, 640, 3), np.uint8))
        except Exception as exc:
            self.depth_error = str(exc)

    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        """Image/video tests use depth from the same tested frame."""
        return self.predict_with_depth(frame_bgr, self.estimate_depth(frame_bgr))

    def estimate_depth(self, frame_bgr: np.ndarray):
        with self._depth_lock:
            if self.depth_error:
                return None
            try:
                from modules.depth_estimator import compute_depth_map
                return compute_depth_map(frame_bgr)
            except Exception as exc:
                self.depth_error = str(exc)
                return None

    def _detect(self, frame_bgr: np.ndarray, depth_map) -> list[Detection]:
        """People (and, for the stock model, knife/scissors) with their depth."""
        from modules.tests.live_spatial_test import yolo_to_detections
        from modules.spatial import CHILD_LABELS

        with self._lock:
            result = self.model.predict(frame_bgr, conf=.2, imgsz=getattr(self, "imgsz", 960),
                                        classes=self.wanted, verbose=False)[0]
            return [d for d in yolo_to_detections(result, depth_map)
                    if d.confidence >= (.35 if d.label in CHILD_LABELS else .2)]

    def predict_with_depth(self, frame_bgr: np.ndarray, depth_map) -> ClassificationResult:
        """Live UI supplies its latest completed map, like the team's demo."""
        if depth_map is not None and depth_map.shape != frame_bgr.shape[:2]:
            depth_map = None
        detections = self._detect(frame_bgr, depth_map)
        score, reason = self.spatial.assess(detections, frame_bgr.shape[:2])
        status = "Relative depth active" if depth_map is not None else "Image-plane distance only; waiting for depth"
        if self.depth_error:
            status = f"Depth unavailable; image-plane distance only: {self.depth_error}"
        return ClassificationResult(
            score_to_level(score), score, detections, reason, depth_status=status,
        )


class HazardSpatialClassifier(SpatialDemoClassifier):
    """Full pipeline: Tze's trained model finds and names the hazard (14 classes),
    the stock COCO model finds the person used as the child reference, depth is
    added to every box, and Ayumi & Nic's SpatialAnalyzer turns hazard type +
    distance to the child into the danger score."""

    imgsz = 640                      # people are large; 640 is enough and much faster on CPU

    def __init__(self, hazard_weights: str) -> None:
        from modules.detector import YOLODetector
        from modules.spatial import CHILD_LABELS

        super().__init__()           # person model + depth model, as in the spatial demo
        self.wanted = [i for i, name in self.names.items() if name in CHILD_LABELS]
        self.detector = YOLODetector(hazard_weights)
        self.mode = f"Trained hazard model: {len(self.detector.names)} classes"
        import torch
        torch.set_num_threads(2)     # loading YOLO resets the CPU pool; keep cores for the video

    def _detect(self, frame_bgr: np.ndarray, depth_map) -> list[Detection]:
        from modules.tests.live_spatial_test import yolo_to_detections

        people = super()._detect(frame_bgr, depth_map)
        with self._lock:
            hazards = yolo_to_detections(self.detector.predict_raw(frame_bgr), depth_map)
        return people + hazards


def get_classifier() -> BaseClassifier:
    """Trained hazard model + spatial analysis when weights/final_best.pt exists,
    otherwise the spatial team's stock-YOLO demo (knife and scissors only)."""
    from modules.detector import find_weights

    weights = find_weights()
    return HazardSpatialClassifier(weights) if weights else SpatialDemoClassifier()
