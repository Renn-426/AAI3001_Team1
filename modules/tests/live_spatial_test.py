"""
Live webcam test of spatial.py + depth_estimator.py using the stock COCO
YOLOv8 model - no trained hazard weights needed.

COCO "person" stands in for the child. COCO "knife" and "scissors" are the
only COCO classes that are also in DANGER_WEIGHTS, so use those as hazards
(or add easier props with --extra-hazards, e.g. "cell phone,cup").

Depth runs in a background thread on the newest frame (it is slow on CPU),
and every frame's boxes are sampled from the latest finished depth map.

Run from the repo root:
    python modules/tests/live_spatial_test.py
    python modules/tests/live_spatial_test.py --no-depth
    python modules/tests/live_spatial_test.py --extra-hazards "cell phone,cup"
    python modules/tests/live_spatial_test.py --show-all --conf 0.15 --model yolov8s.pt
    python modules/tests/live_spatial_test.py --zoom

Keys: q = quit, d = toggle depth view, s = save screenshot
"""
import argparse
import os
import sys
import textwrap
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from modules.classifier import Detection, DangerLevel, score_to_level
from modules.spatial import (
    SpatialAnalyzer, ScoreHold, CHILD_LABELS, DANGER_WEIGHTS, DEFAULT_HAZARD_WEIGHT,
)

WHITE = (255, 255, 255)
GREY = (160, 160, 160)
ZOOM_PADDING = 0.25   # --zoom: extra area around the child, as a share of box size
SAME_OBJECT_IOU = 0.3 # --zoom: overlap above this means both boxes are one object


def level_bgr(level: DangerLevel) -> tuple[int, int, int]:
    """DangerLevel.color is '#rrggbb' (for the UI); OpenCV wants BGR."""
    h = level.color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


class DepthWorker(threading.Thread):
    """Keeps computing depth maps for the newest submitted frame, so the
    webcam view stays smooth while depth (slow on CPU) catches up."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._frame = None
        self.depth_map = None
        self.seconds = None   # time taken by the last depth map
        self.error = None

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def load(self):
        """Call before start(), from the main thread. torchvision is imported
        lazily by both YOLO and the depth model, and importing it from two
        threads at once can fail with a circular ImportError."""
        try:
            import torch
            from modules.depth_estimator import get_depth_pipeline
            get_depth_pipeline(device=0 if torch.cuda.is_available() else None)
        except Exception as e:  # show it on screen instead of killing the test
            self.error = f"{type(e).__name__}: {e}"

    def run(self):
        if self.error:
            return
        from modules.depth_estimator import compute_depth_map

        while True:
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            self.depth_map = compute_depth_map(frame)
            self.seconds = time.perf_counter() - t0

    def status(self) -> str:
        if self.error:
            return f"depth: ERROR {self.error}"
        if self.depth_map is None:
            return "depth: computing first map..."
        return f"depth: {self.seconds:.2f}s per map"


def yolo_to_detections(result, depth_map, offset=(0, 0)) -> list[Detection]:
    """Convert one ultralytics Results into Detections, adding each box's
    0-1 depth when a depth map is available. offset moves boxes from a crop
    back to full-frame pixels."""
    if depth_map is not None:
        from modules.depth_estimator import depth_range, normalize_depth_value, sample_box_depth
        d_range = depth_range(depth_map)

    ox, oy = offset
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        depth = None
        if depth_map is not None:
            depth = normalize_depth_value(sample_box_depth(depth_map, box), depth_map, d_range)
        dets.append(Detection(
            label=result.names[int(b.cls[0])],
            confidence=float(b.conf[0]),
            bbox=box,
            depth=depth,
        ))
    return dets


def box_iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return w * h / (area_a + area_b - w * h)


def zoom_on_child(model, frame, dets, depth_map, hazard_ids, args) -> list[Detection]:
    """Search again for hazards in a crop around the child. Held objects are
    small in the full frame but much larger in the crop. Returns only the
    hazards the full-frame pass missed."""
    children = [d for d in dets if d.label in CHILD_LABELS]
    if not children or not hazard_ids:
        return []
    x1, y1, x2, y2 = max(children, key=lambda d: d.confidence).bbox
    pad_x, pad_y = int((x2 - x1) * ZOOM_PADDING), int((y2 - y1) * ZOOM_PADDING)
    h, w = frame.shape[:2]
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return []

    result = model.predict(frame[cy1:cy2, cx1:cx2], conf=args.conf, imgsz=args.imgsz,
                           classes=hazard_ids, verbose=False)[0]
    return [d for d in yolo_to_detections(result, depth_map, offset=(cx1, cy1))
            if not any(d.label == o.label and box_iou(d.bbox, o.bbox) > SAME_OBJECT_IOU
                       for o in dets)]


def draw(frame, dets, level, score, reason, status):
    color = level_bgr(level)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        is_child = d.label in CHILD_LABELS
        box_color = WHITE if is_child else color
        text = f"{'child' if is_child else d.label} {d.confidence:.2f}"
        if d.depth is not None:
            text += f" d={d.depth:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, text, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # banner: level + score, then the spatial reason wrapped to the frame width
    w = frame.shape[1]
    lines = textwrap.wrap(reason, width=max(20, w // 8))
    banner_h = 34 + 18 * len(lines)
    cv2.rectangle(frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.putText(frame, f"{level.name}  score={score:.2f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 46 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    cv2.putText(frame, status, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def depth_view(depth_map, shape):
    h, w = shape[:2]
    gray = cv2.normalize(np.asarray(depth_map, dtype=np.float32), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(gray, (w, h)), cv2.COLORMAP_INFERNO)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--model", default="yolov8s.pt", help="COCO YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.2, help="minimum hazard confidence")
    parser.add_argument("--child-conf", type=float, default=0.35,
                        help="minimum child confidence (kept higher to avoid false children)")
    parser.add_argument("--zoom", action="store_true",
                        help="search a crop around the child too; finds held objects but is slower")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="seconds to keep the highest recent score (0 = off)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO input size; larger helps small objects, smaller runs faster")
    parser.add_argument("--show-all", action="store_true",
                        help="also draw every other COCO detection in grey (not scored)")
    parser.add_argument("--no-depth", action="store_true", help="planar distance only")
    parser.add_argument("--extra-hazards", default="",
                        help='comma-separated COCO names to treat as hazards, '
                             f'weight {DEFAULT_HAZARD_WEIGHT} (e.g. "cell phone,cup")')
    args = parser.parse_args()

    for name in filter(None, (n.strip() for n in args.extra_hazards.split(","))):
        DANGER_WEIGHTS.setdefault(name.lower().replace(" ", "_"), DEFAULT_HAZARD_WEIGHT)

    model = YOLO(args.model)
    wanted = [i for i, n in model.names.items()
              if n in CHILD_LABELS or n.lower().replace(" ", "_") in DANGER_WEIGHTS]
    wanted_names = {model.names[i] for i in wanted}
    hazard_ids = [i for i in wanted if model.names[i] not in CHILD_LABELS]
    print("Detecting:", ", ".join(sorted(wanted_names)))

    analyzer = SpatialAnalyzer()
    hold = ScoreHold(args.hold)
    worker = None
    if not args.no_depth:
        print("Loading depth model...")
        worker = DepthWorker()
        worker.load()
        worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {args.camera} (try --camera 1).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Webcam: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    show_depth = False
    last_level = None
    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam stopped returning frames.")
            break
        if worker:
            worker.submit(frame.copy())

        depth_map = worker.depth_map if worker else None
        result = model.predict(frame, conf=min(args.conf, args.child_conf), imgsz=args.imgsz,
                               classes=None if args.show_all else wanted, verbose=False)[0]
        dets = yolo_to_detections(result, depth_map)
        others = [d for d in dets if d.label not in wanted_names and d.confidence >= args.conf]
        dets = [d for d in dets if d.label in wanted_names and d.confidence >= (
            args.child_conf if d.label in CHILD_LABELS else args.conf)]
        if args.zoom:
            dets += zoom_on_child(model, frame, dets, depth_map, hazard_ids, args)
        score, reason = analyzer.assess(dets, frame.shape[:2])
        score, reason, held = hold.update(score, reason, time.perf_counter())
        level = score_to_level(score)

        if level != last_level:
            print(f"[{time.strftime('%H:%M:%S')}] {level.name} score={score:.2f} | {reason}")
            last_level = level

        t_now = time.perf_counter()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        status = f"{fps:.0f} fps | " + (worker.status() if worker else "depth: off")
        if held:
            status += " | score held"
        for d in others:  # --show-all: what YOLO sees outside our classes
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
            cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        draw(frame, dets, level, score, reason, status)

        view = frame
        if show_depth and depth_map is not None:
            view = np.hstack([frame, depth_view(depth_map, frame.shape)])
        cv2.imshow("Live spatial test (q quit, d depth, s save)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_depth = not show_depth
        if key == ord("s"):
            path = f"live_test_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(path, view)
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
