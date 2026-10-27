"""
Monocular depth estimation wrapper using Depth Anything V2 (small).

Produces a relative depth map per frame: each pixel gets a value where
HIGHER = CLOSER to the camera (this is Depth Anything's convention —
double check this if you swap models, some output the opposite).

This is RELATIVE depth, not metric (not in actual centimeters/meters).
That's fine for our purpose: we only need to compare "is the hazard at
roughly the same depth as the baby, or very different," not get exact
distances in cm.
"""

import numpy as np
from PIL import Image
from transformers import pipeline

_depth_pipe = None


def get_depth_pipeline(device=None):
    """Lazily load the model once and reuse it across frames (loading
    per-frame would be extremely slow)."""
    global _depth_pipe
    if _depth_pipe is None:
        _depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,  # None = auto (CPU if no GPU found)
        )
    return _depth_pipe


def compute_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    """
    frame_bgr: a single video frame as a numpy array in BGR order (what
    OpenCV gives you from cv2.VideoCapture).
    Returns: a 2D numpy array, same height/width as the frame, with
    relative depth values (higher = closer to camera).
    """
    # transformers/PIL expect RGB, OpenCV gives BGR - convert
    frame_rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(frame_rgb)

    depth_pipe = get_depth_pipeline()
    result = depth_pipe(image)
    depth_map = np.array(result["depth"])  # same H,W as input image

    return depth_map


def sample_box_depth(depth_map: np.ndarray, box, method="median") -> float:
    """
    Get a single representative depth value for a bounding box.

    Using the median over the whole box (not just the center point) is
    more robust - the center pixel might land on a gap/hole/occlusion,
    especially for thin objects like a pen cap.

    Returns a RAW depth value (not yet normalized to 0-1) - use
    normalize_depth_value() below with the frame's min/max to get the
    0-1 value that spatial_analyzer.py expects.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0  # degenerate box, shouldn't normally happen

    region = depth_map[y1:y2, x1:x2]
    if method == "median":
        return float(np.median(region))
    return float(np.mean(region))


def normalize_depth_value(raw_value: float, depth_map: np.ndarray) -> float:
    """
    Scale a raw depth value to 0-1 using THIS FRAME's own min/max.

    Important: depth maps are per-frame relative, not comparable across
    frames or videos. Always normalize against the same frame's depth_map
    that the raw_value was sampled from - never reuse min/max from a
    different frame.
    """
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max - d_min < 1e-6:
        return 0.5  # flat depth map (shouldn't really happen), neutral value
    return (raw_value - d_min) / (d_max - d_min)
