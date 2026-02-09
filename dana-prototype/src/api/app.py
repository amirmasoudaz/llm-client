# src/api/app.py
"""Main FastAPI application for Dana AI Copilot."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import threads, documents, usage, health, retry, enhance
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.services.db import DatabaseService
from src.services.storage import StorageService
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    settings = get_settings()
    
    # Initialize services
    app.state.db = DatabaseService()
    app.state.storage = StorageService()
    
    # Connect to database
    await app.state.db.connect()
    
    yield
    
    # Cleanup on shutdown
    await app.state.db.disconnect()


app = FastAPI(
    title="Dana AI Copilot",
    description="Deep AI agent for academic outreach and funding applications",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(threads.router, prefix="/threads", tags=["Threads"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(usage.router, prefix="/usage", tags=["Usage"])
app.include_router(retry.router, tags=["Jobs"])
app.include_router(enhance.router, prefix="/ai", tags=["AI Enhancement"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Dana AI Copilot",
        "version": "0.1.0",
        "status": "operational"
    }

