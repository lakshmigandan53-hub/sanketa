"""
SANKETA - Static ISL Preprocessing & Landmark Normalization Module
Shared by data collection, training, evaluation, and live inference.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

# Canonical 11-class static ISL vocabulary — exact order is the model's class index order.
# Index 0=HELLO, 1=YES, 2=NO, 3=WATER, 4=FOOD, 5=MILK, 6=TEA,
#       7=BOOK, 8=PEN, 9=PHONE, 10=HELP
STATIC_SIGNS: List[str] = [
    "HELLO",   # 0
    "YES",     # 1
    "NO",      # 2
    "WATER",   # 3
    "FOOD",    # 4
    "MILK",    # 5
    "TEA",     # 6
    "BOOK",    # 7
    "PEN",     # 8
    "PHONE",   # 9
    "HELP",    # 10
]

SIGN_TO_INDEX = {sign: idx for idx, sign in enumerate(STATIC_SIGNS)}
INDEX_TO_SIGN = {idx: sign for idx, sign in enumerate(STATIC_SIGNS)}

NUM_LANDMARKS = 21
NUM_COORDINATES = 3  # (x, y, z)
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDINATES  # 63


def normalize_landmarks(raw_coords: np.ndarray) -> np.ndarray:
    """
    Normalizes 21 MediaPipe hand landmarks (63 numerical features)
    to achieve translation invariance and scale invariance.

    Steps:
    1. Reshapes flat array of 63 elements to (21, 3).
    2. Translation Invariance (Wrist Centering):
       Subtract the wrist position (landmark 0) from all 21 landmarks.
       After this step, the wrist is located exactly at (0.0, 0.0, 0.0).
    3. Scale Invariance (Hand Size Normalization):
       Compute the maximum Euclidean distance from the wrist (0, 0, 0)
       to any of the 20 other hand landmarks.
       Divide all coordinates by this maximum distance (with a small epsilon
       to prevent division by zero).

    Parameters:
    -----------
    raw_coords : np.ndarray
        Array of shape (63,) or (21, 3) containing raw MediaPipe landmark floats.

    Returns:
    --------
    np.ndarray
        Shape (63,), dtype float32, normalized and centered relative to wrist.
    """
    arr = np.asarray(raw_coords, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] == NUM_FEATURES:
        pts = arr.reshape((NUM_LANDMARKS, NUM_COORDINATES))
    elif arr.ndim == 2 and arr.shape == (NUM_LANDMARKS, NUM_COORDINATES):
        pts = arr.copy()
    else:
        raise ValueError(
            f"Expected landmark array of shape (63,) or (21, 3), got shape {arr.shape}"
        )

    # 1. Wrist is landmark index 0
    wrist = pts[0, :].copy()

    # 2. Wrist-relative translation
    centered = pts - wrist

    # 3. Scale normalization using maximum distance from wrist
    distances = np.linalg.norm(centered, axis=1)
    max_dist = float(np.max(distances))

    if max_dist < 1e-6:
        max_dist = 1.0  # Avoid division by zero if all points are collapsed

    normalized = centered / max_dist
    return normalized.flatten().astype(np.float32)


def validate_landmark_vector(vec: np.ndarray) -> Tuple[bool, str]:
    """
    Validates a landmark feature vector.

    Checks:
    - Must be a numpy array
    - Must have 63 elements
    - Must not contain NaN or Inf
    - Must have non-zero variance (not all zeros or corrupted)
    """
    if not isinstance(vec, np.ndarray):
        return False, "Features must be a numpy ndarray."

    if vec.size != NUM_FEATURES:
        return False, f"Feature vector size must be {NUM_FEATURES}, got {vec.size}."

    if np.isnan(vec).any():
        return False, "Feature vector contains NaN values."

    if np.isinf(vec).any():
        return False, "Feature vector contains infinite values."

    if float(np.ptp(vec)) < 1e-5:
        return False, "Feature vector has zero spread (all values identical)."

    return True, "Valid"
