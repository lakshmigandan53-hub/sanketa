"""
Translation router — POST /translate/image

Accepts a multipart/form-data image upload, extracts hand landmarks via
MediaPipe, and classifies the ISL sign using the TF/Keras model.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.response import TranslationErrorResponse, TranslationSuccessResponse
from app.services.mediapipe_service import MediaPipeService
from app.services.sign_recognition import SignRecognitionService
from app.utils.image_processing import decode_image

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons — MediaPipe and ISL model are loaded ONCE when
# the worker process starts.
# ---------------------------------------------------------------------------
_mediapipe_svc = MediaPipeService()
_recognition_svc = SignRecognitionService()

# Accepted upload MIME types
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
)


@router.post(
    "/image",
    summary="Translate an ISL hand sign from an uploaded image",
    response_model=None,
    responses={
        200: {
            "description": "Sign detected, soft failure (no hand / model missing / uncertain)",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "Word recognized",
                            "value": {
                                "success": True,
                                "word": "HELLO",
                                "sign": "HELLO",
                                "confidence": 0.95,
                                "message": "ISL sign recognized successfully",
                            },
                        },
                        "uncertain": {
                            "summary": "Low confidence",
                            "value": {
                                "success": True,
                                "word": "uncertain",
                                "sign": "uncertain",
                                "confidence": 0.42,
                                "message": "Low confidence — ensure hand is clearly visible",
                            },
                        },
                        "no_hand": {
                            "summary": "No hand detected",
                            "value": {"success": False, "message": "No hand detected"},
                        },
                        "no_model": {
                            "summary": "Model not trained yet",
                            "value": {
                                "success": False,
                                "message": "ISL model not found. Run train_model.py after collecting data.",
                            },
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid, empty, or unsupported image file"},
    },
)
async def translate_image(
    file: UploadFile = File(
        ...,
        description="Image containing a hand sign (JPEG / PNG / WebP / BMP / TIFF)",
    ),
) -> JSONResponse:
    """
    Full ISL sign-recognition pipeline:

    1. Validate MIME type.
    2. Read bytes and decode to an OpenCV BGR frame.
    3. Extract 63 hand landmark features with MediaPipe (raw x, y, z).
    4. Classify the gesture with the TF/Keras ISL model.
    5. Return ``{success, sign, confidence, message}`` or ``{success, message}``.

    Soft failures (no hand, model missing) are returned as HTTP 200 with
    ``success: false`` so clients can distinguish them from hard HTTP errors.
    """

    # ------------------------------------------------------------------
    # 1. MIME type validation
    # ------------------------------------------------------------------
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Accepted types: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ------------------------------------------------------------------
    # 2. Read raw bytes
    # ------------------------------------------------------------------
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ------------------------------------------------------------------
    # 3. Decode image with OpenCV
    # ------------------------------------------------------------------
    frame = decode_image(raw_bytes)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not decode the uploaded file as an image. "
                "Make sure it is a valid, uncorrupted image file."
            ),
        )

    # ------------------------------------------------------------------
    # 4. MediaPipe hand landmark extraction (63 raw features)
    # ------------------------------------------------------------------
    landmarks = _mediapipe_svc.extract_landmarks(frame)
    if landmarks is None:
        _recognition_svc.reset_buffer()
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message="No hand detected"
            ).model_dump()
        )

    # ------------------------------------------------------------------
    # 5. Model inference
    # ------------------------------------------------------------------
    result = _recognition_svc.predict(landmarks)
    if not result["success"]:
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message=result["message"]
            ).model_dump()
        )

    # ------------------------------------------------------------------
    # 6. Return successful prediction
    # ------------------------------------------------------------------
    word = result.get("word", result.get("sign", ""))
    pred_type = result.get("type", "static")
    return JSONResponse(
        content=TranslationSuccessResponse(
            success=True,
            word=word,
            sign=word,          # backward-compat alias
            type=pred_type,
            confidence=round(float(result["confidence"]), 4),
            probabilities=result.get("probabilities"),
            model=result.get("model", "isl_model.keras"),
            message=result.get("message", "ISL sign recognized successfully"),
        ).model_dump()
    )


from pydantic import BaseModel, Field
import numpy as np


class LandmarksPayload(BaseModel):
    landmarks: list[float] = Field(..., description="63 landmark coordinates (21 landmarks x 3: x, y, z)")


@router.post(
    "/landmarks",
    summary="Translate ISL hand sign directly from 63 MediaPipe landmark coordinates",
)
async def translate_landmarks(payload: LandmarksPayload) -> JSONResponse:
    if len(payload.landmarks) != 63:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranslationErrorResponse(
                success=False,
                message=f"Expected 63 landmark values, got {len(payload.landmarks)}."
            ).model_dump(),
        )

    arr = np.array(payload.landmarks, dtype=np.float32)
    result = _recognition_svc.predict(arr)
    if not result["success"]:
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message=result["message"]
            ).model_dump()
        )

    word = result.get("word", result.get("sign", ""))
    pred_type = result.get("type", "static")
    return JSONResponse(
        content=TranslationSuccessResponse(
            success=True,
            word=word,
            sign=word,
            type=pred_type,
            confidence=round(float(result["confidence"]), 4),
            probabilities=result.get("probabilities"),
            model=result.get("model", "isl_model.keras"),
            message=result.get("message", "ISL sign recognized successfully"),
        ).model_dump()
    )


class SequencePayload(BaseModel):
    sequence: list[list[float]] = Field(..., description="30 frames x 63 landmark coordinates")


@router.post(
    "/sequence",
    summary="Translate dynamic ISL gesture from a 30-frame sequence",
)
async def translate_sequence(payload: SequencePayload) -> JSONResponse:
    seq = np.array(payload.sequence, dtype=np.float32)
    if seq.shape != (30, 63):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranslationErrorResponse(
                success=False,
                message=f"Expected sequence of shape (30, 63), got {seq.shape}."
            ).model_dump(),
        )

    result = _recognition_svc.predict_sequence(seq)
    if not result["success"]:
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message=result["message"]
            ).model_dump()
        )

    word = result.get("word", "")
    return JSONResponse(
        content=TranslationSuccessResponse(
            success=True,
            word=word,
            sign=word,
            type="dynamic",
            confidence=round(float(result["confidence"]), 4),
            message=result.get("message", "Dynamic ISL sign recognized"),
        ).model_dump()
    )


@router.post(
    "/reset-buffer",
    summary="Reset rolling sequence buffer for dynamic gestures",
)
async def reset_buffer() -> JSONResponse:
    _recognition_svc.reset_buffer()
    return JSONResponse(content={"success": True, "message": "Buffer reset successfully"})


@router.post(
    "/burst",
    summary="Translate dynamic gesture from a burst of image frames",
)
async def translate_burst(files: list[UploadFile] = File(...)) -> JSONResponse:
    if len(files) < 10:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranslationErrorResponse(
                success=False, message=f"Expected sequence of frames, got {len(files)}"
            ).model_dump(),
        )

    landmarks_seq = []
    for file in files:
        raw_bytes = await file.read()
        frame = decode_image(raw_bytes)
        if frame is not None:
            lm = _mediapipe_svc.extract_landmarks(frame)
            if lm is not None:
                landmarks_seq.append(lm)

    if len(landmarks_seq) < 10:
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message="Hand was not visible for enough frames during gesture."
            ).model_dump()
        )

    # Resample or interpolate to exactly 30 frames
    indices = np.linspace(0, len(landmarks_seq) - 1, 30).astype(int)
    resampled_seq = np.array([landmarks_seq[i] for i in indices], dtype=np.float32)

    result = _recognition_svc.predict_sequence(resampled_seq)
    if not result["success"]:
        return JSONResponse(
            content=TranslationErrorResponse(
                success=False, message=result["message"]
            ).model_dump()
        )

    word = result.get("word", "uncertain")
    conf = result.get("confidence", 0.0)
    return JSONResponse(
        content=TranslationSuccessResponse(
            success=True,
            word=word,
            sign=word,
            type="dynamic",
            confidence=round(float(conf), 4),
            message=result.get("message", "Dynamic ISL sign recognized"),
        ).model_dump()
    )


