"""Live camera via streamlit-webrtc.

Why webrtc instead of cv2.VideoCapture: once deployed on the cloud, the server
has no camera. WebRTC streams the *viewer's* browser camera to the server, and
it also works locally.
"""
from __future__ import annotations

import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from modules.classifier import BaseClassifier, ClassificationResult, DangerLevel


class CameraProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.classifier: BaseClassifier | None = None
        self.infer_every_n_frames = 10        # run model every N frames (saves CPU)
        self.draw_overlay = True
        self._frame_count = 0
        self._latest_result: ClassificationResult | None = None
        self._latest_frame = None             # last BGR frame (for Telegram photo)

    # ---- thread-safe accessors used by the Streamlit main thread ----
    @property
    def latest_result(self) -> ClassificationResult | None:
        with self._lock:
            return self._latest_result

    @property
    def latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    # ---- called by webrtc for every frame (runs in a worker thread) ----
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if self.classifier and self._frame_count % self.infer_every_n_frames == 0:
            result = self.classifier.predict(img)
            with self._lock:
                self._latest_result = result

        with self._lock:
            self._latest_frame = img
            result = self._latest_result

        if self.draw_overlay and result is not None:
            img = self._draw(img, result)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @staticmethod
    def _draw(img, result: ClassificationResult):
        # bounding boxes from the spatial-detection part
        for d in result.detections:
            if d.bbox:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # danger banner
        hex_c = result.level.color.lstrip("#")
        rgb = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4))
        bgr = rgb[::-1]
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), bgr, -1)
        cv2.putText(img, f"{result.level.name}  ({result.score:.2f})", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
