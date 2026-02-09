# src/outreach_main.py
"""
Core Funding API

This module handles the email review and sending logic.
Recommender and Reminders are now separate services.

Usage:
    gunicorn src.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:4003
"""

import os
from contextlib import asynccontextmanager
from typing import cast

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
from starlette.datastructures import State
import sentry_sdk
from sentry_sdk import capture_exception
from sentry_sdk.integrations.fastapi import FastApiIntegration
import uvicorn

from src.tools.logger import Logger
from src.api.routers.outreach import router
from src.outreach.logic import OutreachLogic


load_dotenv(find_dotenv(".env"))
version = os.environ.get("VERSION", "0.1.0")

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=1.0,
    integrations=[FastApiIntegration()],
    ignore_errors=[]
)

_, log_config = Logger().create(
    application="api",
    file_name="funding_agent_api_app",
    logger_name="fastapi_app",
    config_only=True
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state = cast(State, app.state)
    app.state.logic = OutreachLogic()

    try:
        yield
    finally:
        pass  # No background tasks to clean up


app = FastAPI(
    title="CanApply Funding Agent",
    description="This is the API for the CanApply Funding Agent",
    version=version,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw = await request.body()
    print("invalid payload received: ", raw.decode(errors="replace"))
    capture_exception(exc)
    sentry_sdk.flush(timeout=2.0)
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    raw = await request.body()
    print("invalid payload received: ", raw.decode(errors="replace"))
    capture_exception(exc)
    sentry_sdk.flush(timeout=2.0)
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal response validation error"},
    )


host = os.environ.get("HOST_API", "127.0.0.1")
port = int(os.environ.get("PORT_API", 4003))


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "CanApply Funding Agent",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": version
    }


app.include_router(router, prefix="/api/v1/funding", tags=["Funding Agent"])


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, log_config=log_config)
