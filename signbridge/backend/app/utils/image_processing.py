"""
Image processing utilities.

Provides a single decode_image() function that converts raw bytes
(from an HTTP upload) into an OpenCV BGR NumPy array.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def decode_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes into an OpenCV BGR frame.

    Parameters
    ----------
    raw_bytes:
        The raw bytes of an uploaded image file (any OpenCV-supported format).

    Returns
    -------
    np.ndarray (HxWx3, uint8, BGR) on success, or ``None`` if decoding fails.
    """
    if not raw_bytes:
        logger.warning("decode_image received empty bytes.")
        return None

    try:
        # Convert bytes → 1-D uint8 array, then decode as an image
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("cv2.imdecode returned None — not a valid image.")
            return None

        return frame

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error decoding image: %s", exc, exc_info=True)
        return None


def resize_frame(
    frame: np.ndarray,
    max_width: int = 640,
    max_height: int = 480,
) -> np.ndarray:
    """
    Optionally resize a frame so that it fits within max_width × max_height
    while preserving the aspect ratio.  Useful for reducing MediaPipe latency
    on very high-resolution inputs.

    Parameters
    ----------
    frame:
        OpenCV BGR image.
    max_width / max_height:
        Upper bounds for the output dimensions.

    Returns
    -------
    Resized (or original if already small enough) BGR frame.
    """
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
