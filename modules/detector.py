"""Object detection interfaces and model skeleton. Owner: Tze."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from modules.classifier import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Accept an HxWx3 uint8 BGR image.

        Return labels, confidence values in [0, 1], and bounding boxes
        as (x1, y1, x2, y2) in the original image's pixel coordinates.
        """


class YOLODetector(BaseDetector):
    """Training/model teammate implements model loading and inference here."""

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        # TODO: Load trained weights once here, rather than on every frame.
        # Agree with the spatial team on labels and child identification.
        raise NotImplementedError("YOLO model loading is not implemented yet.")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        # TODO: Run inference and convert predictions to Detection objects.
        raise NotImplementedError("YOLO detection is not implemented yet.")
