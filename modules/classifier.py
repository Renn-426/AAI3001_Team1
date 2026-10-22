"""Danger-level classification base.

The modelling / spatial-detection teammates only need to subclass
`BaseClassifier` and implement `predict()`. The UI never changes.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


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


@dataclass
class ClassificationResult:
    level: DangerLevel
    score: float                      # 0.0 - 1.0 raw danger score
    detections: list[Detection] = field(default_factory=list)
    reason: str = ""                  # human-readable, goes into the Telegram alert


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


class DummyClassifier(BaseClassifier):
    """Placeholder that returns random scores so the UI can be built/tested.

    Replace with the real model, e.g.:
        class MyModelClassifier(BaseClassifier):
            def __init__(self): self.model = load_model("weights.pt")
            def predict(self, frame_bgr): ...
    """

    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        score = random.random() ** 2          # skewed towards safe
        return ClassificationResult(
            level=score_to_level(score),
            score=score,
            detections=[],
            reason="Dummy classifier (no real model connected)",
        )


def get_classifier() -> BaseClassifier:
    """Single place to swap in the real classifier."""
    return DummyClassifier()
