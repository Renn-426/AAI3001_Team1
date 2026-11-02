"""Object detection with Tze's trained YOLOv8 hazard model. Owner: Tze."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from modules.classifier import Detection

# Where get_classifier() looks for the trained model (first match wins).
# The notebook (Section 6.5) saves it to Drive as "CVDL Model/final_best.pt".
WEIGHT_CANDIDATES = ["weights/final_best.pt", "weights/best.pt", "final_best.pt"]

# The notebook calls class 11 "medicine_syrup"; spatial.py's DANGER_WEIGHTS call it
# "liquid_medicine". Detections use the spatial name so the right weight is applied.
LABEL_ALIASES = {"medicine_syrup": "liquid_medicine"}


def find_weights() -> str | None:
    """HAZARD_MODEL_PATH (env var or Streamlit secret) wins, else the default spots."""
    path = os.getenv("HAZARD_MODEL_PATH")
    if not path:
        try:
            import streamlit as st
            path = st.secrets.get("HAZARD_MODEL_PATH")
        except Exception:
            path = None
    root = Path(__file__).resolve().parents[1]
    for candidate in ([path] if path else WEIGHT_CANDIDATES):
        p = Path(candidate) if Path(candidate).is_absolute() else root / candidate
        if p.is_file():
            return str(p)
    return None


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Accept an HxWx3 uint8 BGR image.

        Return labels, confidence values in [0, 1], and bounding boxes
        as (x1, y1, x2, y2) in the original image's pixel coordinates.
        """


class YOLODetector(BaseDetector):
    """Runs the YOLOv8 weights trained in Child_Safety_Hazard_Detection_YOLOv8.ipynb."""

    def __init__(self, weights_path: str, conf: float = 0.3, imgsz: int | None = None) -> None:
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.model = YOLO(weights_path)                 # loaded once, reused for every frame
        self.conf = conf
        # Same image size the model was trained with (640, or 800 for Experiment 2).
        self.imgsz = imgsz or int(self.model.overrides.get("imgsz", 640))
        self.names = {i: LABEL_ALIASES.get(n, n) for i, n in self.model.names.items()}
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False)  # warm-up

    def predict_raw(self, frame_bgr: np.ndarray):
        """One ultralytics Results object, with class names already aliased."""
        result = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        result.names = self.names
        return result

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.predict_raw(frame_bgr)
        return [
            Detection(self.names[int(c)], float(p), tuple(int(v) for v in xyxy))
            for xyxy, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
        ]
