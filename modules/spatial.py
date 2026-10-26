"""Spatial analysis interfaces and skeleton. Owners: Ayumi & Nic."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modules.classifier import Detection


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
        # TODO: Identify children and hazards using agreed detector labels.
        # TODO: Define the distance method and convert proximity to risk.
        # TODO: Define behaviour for missing detections or bounding boxes.
        raise NotImplementedError("Spatial risk assessment is not implemented yet.")
