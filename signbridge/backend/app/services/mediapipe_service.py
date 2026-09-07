"""
MediaPipe hand-detection service for SANKETA.

Preprocessing pipeline is derived directly from the reference implementation:
  https://github.com/mohamednoorulnaseem/ISL-Recognition-System

Reference preprocessing (app/core/preprocessor.py):
  - Landmarks stored as shape (21, 3) then flattened to (63,)
  - Raw MediaPipe normalized coordinates used directly
  - NO additional wrist-relative or bounding-box normalization

This service matches that pipeline exactly so that any model trained with
the reference code will receive identically shaped and scaled features.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe returns 21 landmarks per hand, each with (x, y, z)
NUM_LANDMARKS = 21
FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63


class MediaPipeService:
    """
    Wrapper around MediaPipe Hands for ISL hand landmark extraction.

    Landmark order matches reference preprocessing exactly:
      [lm0.x, lm0.y, lm0.z, lm1.x, lm1.y, lm1.z, ..., lm20.x, lm20.y, lm20.z]
    """

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        """
        Initialise MediaPipe Hands in static_image_mode=True (correct for
        single uploaded images — no inter-frame tracking required).
        """
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed: bool = False
        logger.info("MediaPipe Hands initialised (max_hands=%d).", max_num_hands)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_landmarks(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the first hand in *bgr_frame* and return its 63-feature vector.

        Preprocessing matches the reference repository exactly:
          1. Convert BGR -> RGB (MediaPipe requirement).
          2. Run MediaPipe Hands.
          3. If no hand found -> return None.
          4. Extract (x, y, z) for each of the 21 landmarks.
          5. Flatten in landmark order: [lm0.x, lm0.y, lm0.z, lm1.x, ...].
          6. Cast to float32.
          NO additional normalization is applied.

        Parameters
        ----------
        bgr_frame : np.ndarray
            OpenCV BGR image (uint8, H×W×3).

        Returns
        -------
        np.ndarray of shape (63,), dtype float32, or None if no hand detected.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            logger.warning("Empty frame passed to extract_landmarks.")
            return None

        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            logger.debug("No hand detected by MediaPipe.")
            return None

        # Use the first detected hand only
        hand_landmarks = results.multi_hand_landmarks[0]
        return self._landmarks_to_array(hand_landmarks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_array(hand_landmarks) -> np.ndarray:
        """
        Convert a MediaPipe NormalizedLandmarkList to a flat float32 array.

        This is the EXACT preprocessing used in the reference repository
        (app/core/preprocessor.py, load_raw_data):

            landmarks = np.load(filepath)       # shape (21, 3)
            landmarks = landmarks.flatten()     # shape (63,)

        MediaPipe's x, y, z are already normalised to [0, 1] relative to the
        image size. No further transformation is applied.
        """
        # Build (21, 3) array in the same order MediaPipe produces landmarks
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )  # shape: (21, 3)

        return coords.flatten()  # shape: (63,)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""
        if not self._closed:
            self._hands.close()
            self._closed = True
            logger.info("MediaPipe Hands closed.")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
