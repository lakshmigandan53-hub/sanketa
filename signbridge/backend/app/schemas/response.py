"""
Pydantic v2 response schemas for SANKETA API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")

    model_config = {
        "json_schema_extra": {
            "example": {"status": "healthy", "service": "SANKETA AI Backend"}
        }
    }


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Response for GET /model/info."""

    model_loaded: bool = Field(..., description="Whether the ISL model is loaded")
    model_path: str = Field(..., description="Expected model file path")
    input_features: int = Field(..., description="Number of input features (63)")
    supported_classes: int = Field(..., description="Number of ISL sign classes")
    classes: List[str] = Field(..., description="List of ISL sign class labels")
    static_model: Optional[Dict[str, Any]] = Field(default=None, description="Static ISL model metadata")
    dynamic_model: Optional[Dict[str, Any]] = Field(default=None, description="Dynamic ISL sequence model metadata")
    pretrained_model: Optional[Dict[str, Any]] = Field(default=None, description="Pretrained image-based model metadata")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "model_loaded": True,
                "model_path": "backend/app/models/isl_static_best.keras + backend/app/models/isl_dynamic_best.keras",
                "input_features": 63,
                "supported_classes": 8,
                "classes": ["HELLO", "HELP", "NO", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES"],
                "static_model": {
                    "status": "loaded",
                    "path": "backend/app/models/isl_static_best.keras",
                    "input_dimensions": "63",
                    "classes": ["HELLO", "NO", "WATER", "YES"],
                    "num_classes": 4,
                },
                "dynamic_model": {
                    "status": "loaded",
                    "path": "backend/app/models/isl_dynamic_best.keras",
                    "input_dimensions": "30x63",
                    "classes": ["HELP", "PLEASE", "SORRY", "THANK_YOU"],
                    "num_classes": 4,
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Translation — success
# ---------------------------------------------------------------------------


class TranslationSuccessResponse(BaseModel):
    """Returned when a hand sign or word is successfully classified."""

    success: bool = Field(default=True, description="Always True for this response")
    word: str = Field(..., description="Predicted ISL word / sign label")
    sign: str = Field(..., description="Alias for 'word' (backward compatibility)")
    type: str = Field(default="static", description="Model type: 'static' or 'dynamic'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-class predicted probabilities",
    )
    model: Optional[str] = Field(
        default="isl_model.keras",
        description="Name of the model that produced the prediction",
    )
    message: str = Field(
        default="ISL sign recognized successfully",
        description="Human-readable result description",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "word": "HELLO",
                "sign": "HELLO",
                "confidence": 0.95,
                "message": "ISL sign recognized successfully",
            }
        }
    }


# ---------------------------------------------------------------------------
# Translation — soft failure (no hand / missing model / prediction error)
# ---------------------------------------------------------------------------


class TranslationErrorResponse(BaseModel):
    """Returned as HTTP 200 when translation cannot complete (soft failure)."""

    success: bool = Field(default=False, description="Always False for this response")
    message: str = Field(..., description="Reason why translation failed")

    model_config = {
        "json_schema_extra": {
            "example": {"success": False, "message": "No hand detected"}
        }
    }
