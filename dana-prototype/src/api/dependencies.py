# src/api/dependencies.py
"""FastAPI dependency injection utilities."""

from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Header, Request, status

from src.services.db import DatabaseService
from src.services.storage import StorageService
from src.services.jobs import JobService
from src.services.events import EventService
from src.config import get_settings, Settings


async def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


async def get_db(request: Request) -> DatabaseService:
    """Get database service from app state."""
    return request.app.state.db


async def get_storage(request: Request) -> StorageService:
    """Get storage service from app state."""
    return request.app.state.storage


async def get_job_service(
    db: Annotated[DatabaseService, Depends(get_db)]
) -> JobService:
    """Get job service with database dependency."""
    return JobService(db)


async def get_event_service() -> EventService:
    """Get event service for SSE/webhooks."""
    return EventService()


async def verify_student_auth(
    x_student_id: Annotated[Optional[str], Header()] = None,
    authorization: Annotated[Optional[str], Header()] = None,
) -> int:
    """
    Verify student authentication.
    
    In production, this will validate the token with the platform backend.
    For now, it extracts student_id from header.
    """
    if not x_student_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Student-ID header"
        )
    
    try:
        student_id = int(x_student_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid student ID format"
        )
    
    # TODO: Validate with platform backend
    # response = await platform_client.verify_auth(authorization, student_id)
    # if not response.valid:
    #     raise HTTPException(status_code=401, detail="Invalid authentication")
    
    return student_id


async def verify_request_access(
    request_id: int,
    student_id: Annotated[int, Depends(verify_student_auth)],
    db: Annotated[DatabaseService, Depends(get_db)],
) -> int:
    """Verify student has access to a funding request."""
    request = await db.get_funding_request(request_id)
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Funding request not found"
        )
    
    if request.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this funding request"
        )
    
    return request_id


# Type aliases for common dependencies
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DBDep = Annotated[DatabaseService, Depends(get_db)]
StorageDep = Annotated[StorageService, Depends(get_storage)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
EventServiceDep = Annotated[EventService, Depends(get_event_service)]
StudentIDDep = Annotated[int, Depends(verify_student_auth)]
RequestAccessDep = Annotated[int, Depends(verify_request_access)]





