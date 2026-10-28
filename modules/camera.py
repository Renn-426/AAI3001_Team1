"""Live camera via streamlit-webrtc.

Why webrtc instead of cv2.VideoCapture: once deployed on the cloud, the server
has no camera. WebRTC streams the *viewer's* browser camera to the server, and
it also works locally.

The model runs on its own background thread: a slow model lowers the
detection rate but never freezes the video.
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from modules.classifier import BaseClassifier, ClassificationResult


def draw_result(img, result: ClassificationResult):
    """Boxes from the spatial/model part + a coloured danger banner.
    Shared by the live camera and the image tester in app.py."""
    img = img.copy()
    for d in result.detections:
        if d.bbox:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    hex_c = result.level.color.lstrip("#")
    bgr = tuple(int(hex_c[i:i + 2], 16) for i in (4, 2, 0))
    s = max(img.shape[1] / 640, 1.0)          # keep the banner readable on large images
    cv2.rectangle(img, (0, 0), (img.shape[1], int(32 * s)), bgr, -1)
    cv2.putText(img, f"{result.level.name} ({result.score:.2f}) {result.reason}"[:80],
                (int(8 * s), int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                (255, 255, 255), max(int(2 * s), 1), cv2.LINE_AA)
    return img


class CameraProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.classifier: BaseClassifier | None = None   # set by app.py
        self.max_infer_fps = 5.0              # cap on model runs per second (saves CPU)
        self.draw_overlay = True
        self._frame_id = 0
        self._latest_frame = None             # last BGR frame (for Telegram photo)
        self._latest_result: ClassificationResult | None = None
        self.infer_fps = 0.0                  # measured model runs per second
        self.latency_ms = 0.0                 # time of the last predict() call
        self.error: str | None = None         # last exception from the model, shown in the UI
        self._stop = threading.Event()
        threading.Thread(target=self._infer_loop, daemon=True).start()

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
        with self._lock:
            self._latest_frame = img
            self._frame_id += 1
            result = self._latest_result
        if self.draw_overlay and result is not None:
            img = draw_result(img, result)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self) -> None:
        self._stop.set()

    # ---- background inference: always uses the newest frame, skips the rest ----
    def _infer_loop(self) -> None:
        last_id, last_done = 0, time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                frame, fid = self._latest_frame, self._frame_id
            if self.classifier is None or frame is None or fid == last_id:
                time.sleep(0.02)
                continue
            start = time.perf_counter()
            try:
                result = self.classifier.predict(frame)
                self.error = None
            except Exception as e:            # a model crash must not kill the video
                self.error = repr(e)
                time.sleep(1.0)
                continue
            with self._lock:
                self._latest_result = result
            last_id = fid

            now = time.perf_counter()
            self.latency_ms = (now - start) * 1000
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / max(now - last_done, 1e-6)
            last_done = now
            wait = start + 1.0 / max(self.max_infer_fps, 0.1) - now
            if wait > 0:
                time.sleep(wait)


# ---------------- uploaded video: same classifier, offline ----------------
@dataclass
class VideoReport:
    timeline: list[tuple[float, ClassificationResult]] = field(default_factory=list)
    worst_t: float = 0.0                        # seconds into the video
    worst_result: ClassificationResult | None = None
    worst_frame: np.ndarray | None = None
    annotated_mp4: bytes | None = None          # H.264, plays in the browser
    seconds: float = 0.0
    truncated: bool = False                     # stopped at max_seconds


def analyse_video(path: str, classifier: BaseClassifier, every_n: int = 5,
                  overlay: bool = True, max_seconds: float = 120, width: int = 640,
                  on_progress=None) -> VideoReport:
    """Run the classifier on every Nth frame and write an annotated copy.

    Frames are resized to `width` first (the model and the boxes then use the
    same coordinates). Frames between two model runs reuse the last result.
    """
    rep = VideoReport()
    src = av.open(path)
    stream = src.streams.video[0]
    fps = float(stream.average_rate or 25)
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    enc = None
    last = None
    try:
        for i, frame in enumerate(src.decode(stream)):
            t = float(frame.time) if frame.time is not None else i / fps
            if t > max_seconds:
                rep.truncated = True
                break
            img = frame.to_ndarray(format="bgr24")
            h = int(img.shape[0] * width / img.shape[1]) // 2 * 2    # H.264 needs even sizes
            img = cv2.resize(img, (width, h))
            if enc is None:
                enc = out.add_stream("libx264", rate=round(fps))
                enc.width, enc.height, enc.pix_fmt = width, h, "yuv420p"
            if i % every_n == 0:
                last = classifier.predict(img)
                rep.timeline.append((t, last))
                if rep.worst_result is None or last.score > rep.worst_result.score:
                    rep.worst_t, rep.worst_result, rep.worst_frame = t, last, img
                if on_progress:
                    on_progress(min(t / min(duration or max_seconds, max_seconds), 1.0))
            shown = draw_result(img, last) if overlay and last else img
            for packet in enc.encode(av.VideoFrame.from_ndarray(shown, format="bgr24")):
                out.mux(packet)
            rep.seconds = t
        if enc is not None:
            for packet in enc.encode():
                out.mux(packet)
    finally:
        out.close()
        src.close()
    rep.annotated_mp4 = buf.getvalue() if enc is not None else None
    return rep
