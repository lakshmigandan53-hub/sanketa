"""
Model information router — GET /model/info
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.response import ModelInfoResponse
from app.services.sign_recognition import SignRecognitionService, _MODEL_DISPLAY_PATH

router = APIRouter()

# Reuse the same singleton that translate.py uses
# Import lazily to avoid circular dependency
_recognition_svc: SignRecognitionService | None = None


def _get_svc() -> SignRecognitionService:
    """Return the shared SignRecognitionService singleton from translate module."""
    from app.api.translate import _recognition_svc as svc  # noqa: PLC0415
    return svc


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get ISL model metadata and status",
)
async def model_info() -> JSONResponse:
    """
    Returns the current ISL model status, expected input shape,
    number of supported classes, and the full class label list.
    """
    svc = _get_svc()
    return JSONResponse(
        content=ModelInfoResponse(
            model_loaded=svc.is_loaded,
            model_path=svc.display_path,
            input_features=63,
            supported_classes=svc.num_classes,
            classes=svc.class_labels,
            static_model=svc.static_info,
            dynamic_model=svc.dynamic_info,
            pretrained_model=None,
        ).model_dump()
    )
