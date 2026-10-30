"""
Combined pipeline: baby detector + hazard detector + SpatialAnalyzer.

Runs both YOLO models per video frame, converts their outputs into the
`Detection` objects SpatialAnalyzer.assess() expects, calls assess(),
and overlays the resulting (score, reason) on the output video.

This reuses spatial_analyzer.py as-is — no risk logic is duplicated here,
only model inference + glue code.

Usage:
    python baby_hazard_pipeline.py \
        --video input.mp4 \
        --out output_risk.mp4 \
        --baby_model yolov8n.pt \
        --hazard_model my_hazard_weights.pt
"""

import argparse
import csv
import sys
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# --- reuse the real SpatialAnalyzer, don't reimplement the risk math ---
sys.path.insert(0, ".")
try:
    # Prefer the team's real Detection class if it's importable here.
    from modules.classifier import Detection  # type: ignore
except ImportError:
    # Fallback stub with the same shape assumed in spatial_analyzer.py
    # (.label: str, .confidence: float, .bbox: (x1, y1, x2, y2) pixels).
    # Safe to use standalone; swap to the real import once this runs
    # inside your team's repo.
    @dataclass
    class Detection:  # type: ignore
        label: str
        confidence: float
        bbox: tuple[float, float, float, float]

from spatial import SpatialAnalyzer, CHILD_LABELS, DANGER_WEIGHTS

ZONE_COLORS = {
    "CRITICAL": (0, 0, 255),
    "WARNING": (0, 140, 255),
    "CAUTION": (0, 255, 255),
    "SAFE": (0, 200, 0),
}


def yolo_results_to_detections(results, conf_thres: float) -> list[Detection]:
    """Convert one ultralytics Results object into a list of Detection."""
    names = results.names
    out = []
    for b in results.boxes:
        conf = float(b.conf[0])
        if conf < conf_thres:
            continue
        cls_id = int(b.cls[0])
        label = names[cls_id]
        box = tuple(b.xyxy[0].tolist())  # (x1, y1, x2, y2)
        out.append(Detection(label=label, confidence=conf, bbox=box))
    return out


def draw_box(frame, box, color, label, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def score_to_zone(score: float) -> str:
    """Map the SpatialAnalyzer's 0-1 score back to a display zone/color.
    Must stay consistent with spatial_analyzer.ZONE_THRESHOLDS logic
    (score = proximity * weight, so this is an approximation for display
    only — the authoritative zone text is already inside `reason`)."""
    if score >= 0.6:
        return "CRITICAL"
    if score >= 0.3:
        return "WARNING"
    if score > 0.0:
        return "CAUTION"
    return "SAFE"


def process_video(video_path, out_path, baby_model_path, hazard_model_path,
                   baby_conf=0.4, hazard_conf=0.3):
    baby_model = YOLO(baby_model_path)
    hazard_model = YOLO(hazard_model_path)
    analyzer = SpatialAnalyzer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    log_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        baby_res = baby_model.predict(frame, verbose=False)[0]
        hazard_res = hazard_model.predict(frame, verbose=False)[0]

        baby_dets = yolo_results_to_detections(baby_res, baby_conf)
        # keep only the actual child/person class from the baby model,
        # in case that model has other classes (eyes, crib, etc.)
        baby_dets = [d for d in baby_dets if d.label.lower() in CHILD_LABELS]

        hazard_dets = yolo_results_to_detections(hazard_res, hazard_conf)
        hazard_dets = [d for d in hazard_dets
                        if d.label.lower().replace(" ", "_") in DANGER_WEIGHTS]

        all_dets = baby_dets + hazard_dets

        score, reason = analyzer.assess(all_dets, (height, width))
        zone = score_to_zone(score)

        # --- draw boxes ---
        for d in baby_dets:
            draw_box(frame, d.bbox, (255, 255, 255), "baby/person")
        for d in hazard_dets:
            draw_box(frame, d.bbox, ZONE_COLORS[zone], d.label)

        # --- banner ---
        cv2.rectangle(frame, (0, 0), (width, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"Risk: {zone} (score={score:.2f})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLORS[zone], 2)

        writer.write(frame)
        log_rows.append((frame_idx, round(t_sec, 2), zone, score, reason))

    cap.release()
    writer.release()
    return log_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="output_risk.mp4")
    parser.add_argument("--baby_model", required=True)
    parser.add_argument("--hazard_model", required=True)
    parser.add_argument("--baby_conf", type=float, default=0.4)
    parser.add_argument("--hazard_conf", type=float, default=0.3)
    args = parser.parse_args()

    rows = process_video(
        args.video, args.out, args.baby_model, args.hazard_model,
        baby_conf=args.baby_conf, hazard_conf=args.hazard_conf,
    )

    csv_path = args.out.rsplit(".", 1)[0] + "_timeline.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "zone", "score", "reason"])
        w.writerows(rows)

    print(f"Done. Annotated video: {args.out}")
    print(f"Risk timeline CSV: {csv_path}")
