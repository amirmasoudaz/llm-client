# src/recommender_main.py
"""
Recommender Service Entry Point with Hot-Swap Indexing

This module implements a hot-swap pattern for the recommender index:
1. On startup: Load existing index from Qdrant (fast, read-only)
2. Immediately mark service as READY and start serving requests
3. Kick off background indexing task to update the index
4. When indexing completes: hot-swap the new index in place of the old one

This ensures that:
- Startup is fast (no blocking on indexing)
- Requests are served immediately with the existing index
- New data is picked up in background without downtime

Designed to be run with Gunicorn's --preload flag:
    gunicorn src.recommender_main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --preload --bind 0.0.0.0:4004

The TagRecommender is loaded at module level (before forking) so all workers
share the same memory-mapped index data via copy-on-write.
"""

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, cast, Any

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

from src.tools.logger import Logger
from src.api.routers.recommender import router
from src.recommender.logic import RecommenderLogic


load_dotenv(find_dotenv(".env"))
version = os.environ.get("VERSION", "0.1.0")

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=1.0,
    integrations=[FastApiIntegration()],
    ignore_errors=[]
)

_, log_config = Logger().create(
    application="recommender_api",
    file_name="recommender_api_app",
    logger_name="recommender_app",
    config_only=True
)

# ==============================================================================
# MODULE-LEVEL INDEX LOADING (for Gunicorn --preload)
# ==============================================================================
# This runs BEFORE forking workers, so the index is loaded once and shared.
# We use load_from_store() for FAST startup - no indexing, just load existing data.
print("Loading TagRecommender index at module level (preload phase)...")
_PRELOADED_RECOMMENDER: Optional[RecommenderLogic] = None
_PRELOAD_SUCCESS: bool = False

try:
    _PRELOADED_RECOMMENDER = RecommenderLogic(lazy_init=True)
    _PRELOAD_SUCCESS = _PRELOADED_RECOMMENDER.load_from_store()
    if _PRELOAD_SUCCESS:
        print(f"TagRecommender preloaded successfully with {_PRELOADED_RECOMMENDER.n_docs} documents (fast load, no indexing).")
    else:
        print("No existing index found. Service will start but wait for background indexing.")
except Exception as e:
    print(f"Failed to preload TagRecommender: {e}")
    capture_exception(e)
# ==============================================================================


async def background_reindex_and_swap(app: FastAPI, force_rebuild: bool = False) -> None:
    """
    Background task that indexes new data and hot-swaps the recommender.
    
    This runs independently of request handling, so requests continue to be
    served with the old index while reindexing happens.
    """
    # Wait a moment for the server to fully start up before beginning heavy work
    await asyncio.sleep(2.0)
    
    async with app.state.recommender_lock:
        if app.state.recommender_status["building"]:
            print("Reindex already in progress, skipping.")
            return
        app.state.recommender_status.update({
            "building": True,
            "started_at": time.time(),
            "last_error": "OK",
        })

    print("Background reindexing started (server is accepting requests)...")
    try:
        def build_new_index():
            """Build a fresh recommender with full indexing (runs in thread)."""
            r = RecommenderLogic(lazy_init=True)
            r.prep_data(force_rebuild=force_rebuild)
            return r

        # Run the heavy indexing work in a thread to not block the event loop
        new_rec: RecommenderLogic = await asyncio.to_thread(build_new_index)
        
        # Hot-swap: atomically replace the old recommender with the new one
        async with app.state.recommender_lock:
            old_count = app.state.recommender.n_docs if app.state.recommender else 0
            app.state.recommender = new_rec
            app.state.recommender_status.update({
                "ready": True,
                "version": app.state.recommender_status["version"] + 1,
            })
        
        print(
            f"Hot-swap complete! Recommender updated: {old_count} -> {new_rec.n_docs} documents "
            f"(version {app.state.recommender_status['version']})"
        )

    except Exception as e:
        print(f"Background reindexing failed: {e}")
        capture_exception(e)
        async with app.state.recommender_lock:
            app.state.recommender_status["last_error"] = str(e)
    finally:
        async with app.state.recommender_lock:
            app.state.recommender_status.update({
                "building": False,
                "finished_at": time.time(),
            })


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state = cast(State, app.state)
    app.state.recommender_lock = asyncio.Lock()
    app.state.reindex_jobs: dict[str, dict[str, Any]] = {}

    # Use the preloaded recommender if available
    if _PRELOADED_RECOMMENDER is not None and _PRELOAD_SUCCESS:
        app.state.recommender = _PRELOADED_RECOMMENDER
        app.state.recommender_status = {
            "ready": True,
            "building": False,
            "started_at": None,
            "finished_at": time.time(),
            "version": 1,
            "last_error": "OK",
            "current_job_id": None,
            "load_mode": "fast_preload",
        }
        print("Using preloaded TagRecommender (fast load). Starting background reindex...")
        
        # Immediately kick off background reindexing to pick up any new data
        app.state.indexer_task = asyncio.create_task(
            background_reindex_and_swap(app, force_rebuild=False)
        )

    elif _PRELOADED_RECOMMENDER is not None:
        # Preload object exists but no data was loaded (empty store)
        # Still use it but mark as not ready until indexing completes
        app.state.recommender = _PRELOADED_RECOMMENDER
        app.state.recommender_status = {
            "ready": False,  # Not ready until first index completes
            "building": False,
            "started_at": None,
            "finished_at": None,
            "version": 0,
            "last_error": "OK",
            "current_job_id": None,
            "load_mode": "empty_store_reindex",
        }
        print("Empty store detected. Starting initial indexing...")
        
        app.state.indexer_task = asyncio.create_task(
            background_reindex_and_swap(app, force_rebuild=False)
        )

    else:
        # Fallback: create new recommender in lifespan (for single-worker/dev mode)
        app.state.recommender = None
        app.state.recommender_status = {
            "ready": False,
            "building": False,
            "started_at": None,
            "finished_at": None,
            "version": 0,
            "last_error": "OK",
            "current_job_id": None,
            "load_mode": "lifespan_fallback",
        }
        print("No preloaded recommender. Creating in lifespan with background indexing...")
        
        # Create a recommender and try fast load, then reindex
        async def init_and_reindex():
            rec = RecommenderLogic(lazy_init=True)
            # Try fast load first
            loaded = await asyncio.to_thread(rec.load_from_store)
            if loaded:
                async with app.state.recommender_lock:
                    app.state.recommender = rec
                    app.state.recommender_status.update({
                        "ready": True,
                        "version": 1,
                    })
                print(f"Lifespan fast load complete with {rec.n_docs} documents.")
            
            # Either way, kick off background reindex
            await background_reindex_and_swap(app, force_rebuild=False)
        
        app.state.indexer_task = asyncio.create_task(init_and_reindex())

    # Server is now ready to accept requests
    n_docs = app.state.recommender.n_docs if app.state.recommender else 0
    print(f"=" * 60)
    print(f"Recommender API ready! Serving {n_docs} documents.")
    print(f"Server listening on http://{host}:{port}")
    print(f"Docs available at http://{host}:{port}/docs")
    print(f"=" * 60)
    
    try:
        yield
    finally:
        print("Shutting down recommender service...")
        task: Optional[asyncio.Task] = getattr(app.state, "indexer_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(
    title="CanApply Funding Recommender Service",
    description="Tag recommendation and professor ranking API",
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
port = int(os.environ.get("PORT_RECOMMENDER_API", 4004))


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "CanApply Funding Recommender Service",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": version
    }


@app.get("/api/v1/status", include_in_schema=True, tags=["Status"])
async def recommender_status():
    """Get the current status of the recommender index."""
    status = dict(app.state.recommender_status)
    status["n_docs"] = app.state.recommender.n_docs if app.state.recommender else 0
    return status


app.include_router(router, prefix="/api/v1/funding", tags=["Recommender"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_config=log_config)
