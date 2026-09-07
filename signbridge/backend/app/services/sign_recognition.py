"""
Sign Recognition Service for SANKETA.

Supports dual-model recognition:
1. Static Model:
   - Evaluates the latest 63-feature MediaPipe landmark vector.
   - Supports 11 static ISL signs:
     HELLO, YES, NO, WATER, FOOD, MILK, TEA, BOOK, PEN, PHONE, HELP
   - Class order is loaded from app/models/labels.txt (one per line).
2. Dynamic Model:
   - Maintains a rolling 30-frame sequence buffer.
   - When 30 frames are ready, evaluates the (30, 63) trajectory with LSTM/GRU.
   - Classes: THANK_YOU, PLEASE, SORRY, HELP
"""

from __future__ import annotations

import collections
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).parent.parent
_MODELS_DIR = _APP_DIR / "models"

_LABELS_TXT_PATH = _MODELS_DIR / "labels.txt"
_LABELS_JSON_PATH = _MODELS_DIR / "labels.json"
_ISL_MODEL_PATH = _MODELS_DIR / "isl_model.keras"
_ISL_MODEL_H5_PATH = _MODELS_DIR / "isl_model.h5"

_STATIC_MODEL_BEST_PATH = _MODELS_DIR / "isl_static_best.keras"
_STATIC_MODEL_V2_PATH = _MODELS_DIR / "static_model_v2.keras"
_ISL_MODEL_TF212_PATH = _MODELS_DIR / "isl_model_tf212.h5"

_STATIC_MODEL_PATH = (
    _ISL_MODEL_TF212_PATH          # TF 2.12-compatible H5 (converted from Keras 3.x)
    if _ISL_MODEL_TF212_PATH.exists()
    else (
        _ISL_MODEL_PATH
        if _ISL_MODEL_PATH.exists()
        else (_ISL_MODEL_H5_PATH if _ISL_MODEL_H5_PATH.exists() else _STATIC_MODEL_BEST_PATH)
    )
)

_DYNAMIC_MODEL_BEST_PATH = _MODELS_DIR / "isl_dynamic_best.keras"
_DYNAMIC_MODEL_V2_PATH = _MODELS_DIR / "dynamic_model_v2.keras"
_DYNAMIC_MODEL_PATH = (
    _DYNAMIC_MODEL_BEST_PATH
    if _DYNAMIC_MODEL_BEST_PATH.exists()
    else (_DYNAMIC_MODEL_V2_PATH if _DYNAMIC_MODEL_V2_PATH.exists() else _MODELS_DIR / "dynamic_model.keras")
)

_STATIC_CLASSES_BEST_PATH = _MODELS_DIR / "static_classes_best.json"
_STATIC_CLASSES_V2_PATH = _MODELS_DIR / "static_classes_v2.json"
_STATIC_CLASSES_PATH = (
    _LABELS_JSON_PATH
    if _LABELS_JSON_PATH.exists()
    else (_STATIC_CLASSES_BEST_PATH if _STATIC_CLASSES_BEST_PATH.exists() else _MODELS_DIR / "static_classes.json")
)

_DYNAMIC_CLASSES_BEST_PATH = _MODELS_DIR / "dynamic_classes_best.json"
_DYNAMIC_CLASSES_V2_PATH = _MODELS_DIR / "dynamic_classes_v2.json"
_DYNAMIC_CLASSES_PATH = (
    _DYNAMIC_CLASSES_BEST_PATH
    if _DYNAMIC_CLASSES_BEST_PATH.exists()
    else (_DYNAMIC_CLASSES_V2_PATH if _DYNAMIC_CLASSES_V2_PATH.exists() else _MODELS_DIR / "dynamic_classes.json")
)

_COMBINED_CLASSES_PATH = _MODELS_DIR / "class_names.json"
_LEGACY_MODEL_PATH = _MODELS_DIR / "isl_model.keras"

_DEFAULT_CONFIDENCE_THRESHOLD = float(
    os.getenv("ISL_CONFIDENCE_THRESHOLD", "0.70")
)
_DEFAULT_MOTION_THRESHOLD = float(
    os.getenv("ISL_MOTION_THRESHOLD", "0.07")
)


def normalize_landmarks(raw_coords: np.ndarray) -> np.ndarray:
    """
    Applies wrist-centering and hand-scale normalization.
    Matches the exact preprocessing used during training.
    """
    pts = raw_coords.reshape((21, 3))
    wrist = pts[0, :].copy()
    centered = pts - wrist
    distances = np.linalg.norm(centered, axis=1)
    max_dist = float(np.max(distances))
    if max_dist < 1e-6:
        max_dist = 1.0
    return (centered / max_dist).flatten().astype(np.float32)


def _load_class_list(path: Path, fallback: list[str]) -> list[str]:
    # 1. Check if labels.txt exists first (canonical plain-text class order)
    if _LABELS_TXT_PATH.exists():
        try:
            with open(_LABELS_TXT_PATH, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", _LABELS_TXT_PATH, exc)

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            labels = data.get("classes", [])
            if labels:
                return labels
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
    return fallback


class SignRecognitionService:
    """Dual-model (Static + Dynamic) ISL classification service with Motion Gating."""

    def __init__(
        self,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        motion_threshold: float = _DEFAULT_MOTION_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._motion_threshold = motion_threshold
        self._static_model: Optional[Any] = None
        self._dynamic_model: Optional[Any] = None
        self._static_model_path_used: Path = _STATIC_MODEL_PATH

        # All 11 target static classes (matches training order)
        _STATIC_FALLBACK = [
            "HELLO", "YES", "NO", "WATER", "FOOD",
            "MILK", "TEA", "BOOK", "PEN", "PHONE", "HELP",
        ]
        self._static_classes: List[str] = _load_class_list(
            _STATIC_CLASSES_PATH, _STATIC_FALLBACK
        )
        self._dynamic_classes: List[str] = _load_class_list(
            _DYNAMIC_CLASSES_PATH, ["HELP", "PLEASE", "SORRY", "THANK_YOU"]
        )

        # Rolling buffer for continuous webcam frames (last 30 valid landmark vectors)
        self._rolling_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=30)
        self._last_frame_time: float = 0.0

        self._load_models()

    def _load_models(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed.")
            return

        # 1. Load Static Model (prefer newly trained isl_model.keras)
        static_candidates = [
            _ISL_MODEL_TF212_PATH,      # TF 2.12-compatible H5 (FIRST — avoids Keras 3.x compat issues)
            _ISL_MODEL_PATH,
            _ISL_MODEL_H5_PATH,
            _STATIC_MODEL_V2_PATH,
            _STATIC_MODEL_BEST_PATH,
            _LEGACY_MODEL_PATH,
        ]
        for candidate in static_candidates:
            if candidate.exists():
                try:
                    self._static_model = tf.keras.models.load_model(str(candidate), compile=False)
                    self._static_model_path_used = candidate
                    logger.info("Static ISL model successfully loaded from '%s' (%s classes).", candidate.name, len(self._static_classes))
                    break
                except Exception as e:
                    logger.warning("Candidate static model '%s' could not be loaded: %s", candidate.name, e)

        # 2. Load Dynamic Model (if available)
        dynamic_candidates = [
            _DYNAMIC_MODEL_PATH,
            _DYNAMIC_MODEL_BEST_PATH,
            _DYNAMIC_MODEL_V2_PATH,
        ]
        for candidate in dynamic_candidates:
            if candidate.exists():
                try:
                    self._dynamic_model = tf.keras.models.load_model(str(candidate), compile=False)
                    logger.info("Dynamic ISL model loaded from '%s' (%s).", candidate.name, self._dynamic_classes)
                    break
                except Exception as e:
                    logger.debug("Candidate dynamic model '%s' not loaded: %s", candidate.name, e)

    def reset_buffer(self) -> None:
        """Clears the dynamic sequence rolling buffer."""
        self._rolling_buffer.clear()
        self._last_frame_time = 0.0

    def predict(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Runs dual-model inference with motion-energy gating and temporal continuity.
        - When hand is stationary (motion < 0.07), evaluates Static Dense model.
        - When hand exhibits continuous motion across 30 frames (motion >= 0.07), evaluates Dynamic BiLSTM model.
        - Resets buffer if time between frames > 600ms (prevents stitching slow snapshots into a false sequence).
        - If confidence is below threshold, returns 'uncertain'.
        """
        import time
        now = time.time()
        if self._last_frame_time > 0.0 and (now - self._last_frame_time) > 0.6:
            self._rolling_buffer.clear()
        self._last_frame_time = now

        if self._static_model is None and self._dynamic_model is None:
            return {
                "success": False,
                "message": "ISL models not loaded. Train the models first.",
            }

        if landmarks.ndim != 1 or landmarks.shape[0] != 63:
            return {
                "success": False,
                "message": f"Landmark vector must have shape (63,), got {landmarks.shape}.",
            }

        # Add valid frame to rolling buffer
        self._rolling_buffer.append(landmarks.copy())

        # Compute motion displacement across the rolling buffer
        motion_disp = 0.0
        if len(self._rolling_buffer) >= 2:
            buf_arr = np.array(list(self._rolling_buffer), dtype=np.float32)
            motion_disp = float(np.max(np.linalg.norm(buf_arr - buf_arr[0], axis=1)))

        # --- 1. Static Pose Branch (Evaluate current hand shape) ---
        stat_word = "uncertain"
        stat_conf = 0.0
        stat_probs: Dict[str, float] = {}
        if self._static_model is not None:
            norm_landmarks = normalize_landmarks(landmarks)
            inp = np.expand_dims(norm_landmarks, axis=0)  # (1, 63)
            preds = self._static_model.predict(inp, verbose=0)[0]
            s_idx = int(np.argmax(preds))
            stat_conf = float(preds[s_idx])
            stat_word = self._static_classes[s_idx] if s_idx < len(self._static_classes) else "uncertain"
            stat_probs = {
                cls: round(float(preds[i]), 4)
                for i, cls in enumerate(self._static_classes)
                if i < len(preds)
            }

            if stat_conf >= self._confidence_threshold:
                return {
                    "success": True,
                    "word": stat_word,
                    "sign": stat_word,
                    "type": "static",
                    "confidence": round(stat_conf, 4),
                    "probabilities": stat_probs,
                    "model": "isl_model.keras",
                    "message": f"Static ISL sign recognized ({stat_word})",
                }

        # --- 2. Dynamic Gesture Branch (Only when static pose is unconfident AND significant motion occurred) ---
        if len(self._rolling_buffer) == 30 and motion_disp >= 0.22 and self._dynamic_model is not None:
            inp_seq = np.expand_dims(buf_arr, axis=0)  # (1, 30, 63)
            d_preds = self._dynamic_model.predict(inp_seq, verbose=0)[0]
            d_idx = int(np.argmax(d_preds))
            dyn_conf = float(d_preds[d_idx])
            dyn_word = self._dynamic_classes[d_idx] if d_idx < len(self._dynamic_classes) else "uncertain"
            dyn_probs = {
                cls: round(float(d_preds[i]), 4)
                for i, cls in enumerate(self._dynamic_classes)
                if i < len(d_preds)
            }

            if dyn_conf >= self._confidence_threshold:
                return {
                    "success": True,
                    "word": dyn_word,
                    "sign": dyn_word,
                    "type": "dynamic",
                    "confidence": round(dyn_conf, 4),
                    "probabilities": dyn_probs,
                    "model": "dynamic_model.keras",
                    "message": f"Dynamic ISL sign recognized ({dyn_word})",
                }

        # --- 3. Low Confidence / Uncertain Result (Below 70% threshold) ---
        best_candidate = stat_word if stat_word != "uncertain" else ""
        return {
            "success": True,
            "word": "uncertain",
            "sign": "uncertain",
            "type": "static",
            "confidence": round(stat_conf, 4),
            "probabilities": stat_probs,
            "model": "isl_model.keras",
            "message": f"Uncertain sign (confidence {stat_conf:.0%} < 70%). Candidate: '{best_candidate}'" if best_candidate else "Uncertain sign (confidence < 70%)",
        }

    def predict_sequence(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Runs sequence inference directly on a (30, 63) array.
        """
        if self._dynamic_model is None:
            return {
                "success": False,
                "message": "Dynamic ISL model is not loaded.",
            }

        if sequence.shape != (30, 63):
            return {
                "success": False,
                "message": f"Expected sequence of shape (30, 63), got {sequence.shape}.",
            }

        inp_seq = np.expand_dims(sequence.astype(np.float32), axis=0)
        preds = self._dynamic_model.predict(inp_seq, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        word = self._dynamic_classes[idx] if idx < len(self._dynamic_classes) else "UNKNOWN"

        if conf < self._confidence_threshold:
            return {
                "success": True,
                "word": "uncertain",
                "sign": "uncertain",
                "type": "dynamic",
                "confidence": round(conf, 4),
                "message": f"Low confidence dynamic prediction ({conf:.0%}). Best guess: '{word}'",
            }

        return {
            "success": True,
            "word": word,
            "sign": word,
            "type": "dynamic",
            "confidence": round(conf, 4),
            "message": f"Dynamic ISL sign recognized ({word})",
        }

    @property
    def is_loaded(self) -> bool:
        return self._static_model is not None or self._dynamic_model is not None

    @property
    def static_model_loaded(self) -> bool:
        return self._static_model is not None

    @property
    def dynamic_model_loaded(self) -> bool:
        return self._dynamic_model is not None

    @property
    def class_labels(self) -> List[str]:
        return sorted(list(set(self._static_classes + self._dynamic_classes)))

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def display_path(self) -> str:
        parts = []
        if self._static_model is not None:
            parts.append(f"backend/app/models/{_STATIC_MODEL_PATH.name}")
        if self._dynamic_model is not None:
            parts.append(f"backend/app/models/{_DYNAMIC_MODEL_PATH.name}")
        return " + ".join(parts) if parts else "backend/app/models/static_model_v2.keras"

    @property
    def buffer_length(self) -> int:
        return len(self._rolling_buffer)

    @property
    def static_info(self) -> Dict[str, Any]:
        return {
            "status": "loaded" if self._static_model is not None else "not_loaded",
            "path": f"backend/app/models/{_STATIC_MODEL_PATH.name}",
            "input_dimensions": "63",
            "classes": self._static_classes,
            "num_classes": len(self._static_classes),
        }

    @property
    def dynamic_info(self) -> Dict[str, Any]:
        return {
            "status": "loaded" if self._dynamic_model is not None else "not_loaded",
            "path": f"backend/app/models/{_DYNAMIC_MODEL_PATH.name}",
            "input_dimensions": "30x63",
            "classes": self._dynamic_classes,
            "num_classes": len(self._dynamic_classes),
        }


_MODEL_DISPLAY_PATH = "backend/app/models/isl_static_best.keras"
