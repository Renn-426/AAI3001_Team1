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

# Share of each box's width/height used for depth, kept around its centre.
# Box corners are usually background, which skews the median.
DEPTH_SAMPLE_FRACTION = 0.5


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


def depth_sample_box(box, fraction=DEPTH_SAMPLE_FRACTION) -> tuple[int, int, int, int]:
    """The central part of a box that depth is actually read from
    (x1, y1, x2, y2). Always at least 1 pixel wide and tall."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) * fraction / 2, 1.0)
    half_h = max((y2 - y1) * fraction / 2, 1.0)
    return (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))


def sample_box_depth(depth_map: np.ndarray, box, method="median") -> float:
    """
    Get a single representative depth value for a bounding box.

    Using the median over the central part of the box (not just the
    center point) is more robust - the center pixel might land on a
    gap/hole/occlusion, especially for thin objects like a pen cap, while
    the box corners are often background.

    Returns a RAW depth value (not yet normalized to 0-1) - use
    normalize_depth_value() below with the frame's depth range to get the
    0-1 value that spatial_analyzer.py expects.
    """
    x1, y1, x2, y2 = depth_sample_box(box)
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0  # degenerate box, shouldn't normally happen

    region = depth_map[y1:y2, x1:x2]
    if method == "median":
        return float(np.median(region))
    return float(np.mean(region))


def depth_range(depth_map: np.ndarray) -> tuple[float, float]:
    """The frame's 2nd-98th percentile, so a few extreme pixels can't
    stretch the scale and make values jump between frames."""
    d_min, d_max = np.percentile(depth_map, (2, 98))
    return float(d_min), float(d_max)


def normalize_depth_value(raw_value: float, depth_map: np.ndarray, d_range=None) -> float:
    """
    Scale a raw depth value to 0-1 using THIS FRAME's own depth range.

    Important: depth maps are per-frame relative, not comparable across
    frames or videos. Always normalize against the same frame's depth_map
    that the raw_value was sampled from - never reuse a range from a
    different frame. Pass d_range from depth_range() to avoid recomputing
    it for every box in a frame.
    """
    d_min, d_max = d_range or depth_range(depth_map)
    if d_max - d_min < 1e-6:
        return 0.5  # flat depth map (shouldn't really happen), neutral value
    return float(np.clip((raw_value - d_min) / (d_max - d_min), 0.0, 1.0))
