# src/api/logic.py
"""
Core Funding API Router

Handles email review and sending logic.
Recommender endpoints are now in logic.py.
"""

import sentry_sdk

from fastapi import APIRouter, HTTPException, Depends

from src.outreach.logic import OutreachLogic
from src.api.schemas.outreach import (
    ReviewResponse,
    SendResponse
)
from src.db.session import DB


router = APIRouter()


def get_logic() -> OutreachLogic:
    return OutreachLogic()


@router.get("/healthz/outreach")
async def _get_health():
    try:
        ok = await DB.fetch_one("SELECT 1")
        db_ok = "OK" if ok is not None else "No response"
    except Exception as e:
        db_ok = str(e)
    return {"status": True, "db": db_ok}


@router.get("/{funding_id}/review", response_model=ReviewResponse)
async def _get_chat(funding_id: int, logic: OutreachLogic = Depends(get_logic)):
    try:
        result = await logic.get_review(funding_id=funding_id)
        if not result.get("status", False):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error in /{funding_id}/review endpoint: {e}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{funding_id}/send", response_model=SendResponse)
async def _get_suggestion(funding_id: int, logic: OutreachLogic = Depends(get_logic)):
    try:
        result = await logic.get_send(funding_request_id=funding_id)
        if not result.get("status", False):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error in /{funding_id}/send endpoint: {e}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))
