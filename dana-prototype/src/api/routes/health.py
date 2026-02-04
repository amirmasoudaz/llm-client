# src/api/routes/health.py
"""Health check endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import DBDep, SettingsDep


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: DBDep,
    settings: SettingsDep,
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    db_status = "connected" if await db.is_connected() else "disconnected"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="0.1.0",
        database=db_status,
        environment=settings.app_env,
    )


@router.get("/ready")
async def readiness_check(db: DBDep) -> dict:
    """
    Readiness check for Kubernetes/orchestration.
    
    Returns 200 if the service is ready to accept traffic.
    """
    is_connected = await db.is_connected()
    return {"ready": is_connected}


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes/orchestration.
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}





