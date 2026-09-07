"""
Health-check router — GET /health
"""

from fastapi import APIRouter
from app.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Returns the service status. Useful for load-balancer / uptime probes."""
    return HealthResponse(status="healthy", service="SANKETA AI Backend")
