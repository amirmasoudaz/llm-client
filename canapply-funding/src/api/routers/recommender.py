# src/api/logic.py

from src.config import settings
import time
import asyncio
from typing import List, Dict

from fastapi import APIRouter, HTTPException, Depends, Query, Request
import uuid
import sentry_sdk

from src.api.schemas.recommender import (
    QueryResponse,
    RecommendResponse,
    RecommendRequest
)
from src.recommender.logic import RecommenderLogic
from src.db.session import DB


DEFAULT_TOP_K = settings.DEFAULT_TOP_K

router = APIRouter()


def get_recommender(request: Request) -> RecommenderLogic:
    rec = getattr(request.app.state, "recommender", None)
    if rec is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return rec


@router.get("/healthz/recommender")
async def _get_health():
    try:
        ok = await DB.fetch_one("SELECT 1")
        db_ok = "OK" if ok is not None else "No response"
    except Exception as e:
        db_ok = str(e)
    rec_status = "ready"  # If we get here, recommender is loaded
    return {"status": True, "db": db_ok, "recommender": rec_status}


@router.post("/reindex")
async def _post_reindex(
    request: Request,
    force_rebuild: bool = Query(False, description="Force rebuild of the index"),
):
    async with request.app.state.recommender_lock:
        st = request.app.state.recommender_status
        jobs = getattr(request.app.state, "reindex_jobs", {})
        if st.get("building"):
            current_job = st.get("current_job_id")
            job_info = jobs.get(current_job) if current_job else None
            return {
                "status": True,
                "message": "Reindex already in progress",
                "current_job_id": current_job,
                "job_status": job_info,
            }

        job_id = str(uuid.uuid4())
        now = time.time()
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "force_rebuild": bool(force_rebuild),
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "metrics": None,
        }
        st.update({
            "building": True,
            "started_at": now,
            "last_error": None,
            "current_job_id": job_id,
        })

        async def build_and_swap(current_job_id: str):
            try:
                jobs[current_job_id]["status"] = "building"
                jobs[current_job_id]["started_at"] = time.time()

                def build():
                    r = RecommenderLogic(lazy_init=True)
                    r.prep_data(force_rebuild=force_rebuild)
                    return r

                new_rec = await asyncio.to_thread(build)
                async with request.app.state.recommender_lock:
                    request.app.state.recommender = new_rec
                    st.update({"ready": True, "version": st["version"] + 1})
                jobs[current_job_id]["status"] = "completed"
                jobs[current_job_id]["finished_at"] = time.time()
                jobs[current_job_id]["metrics"] = {
                    "doc_count": new_rec.n_docs,
                    "version": st["version"],
                }
                print("Recommender indexed and swapped successfully.")
            except Exception as e:
                sentry_sdk.capture_exception(e)
                async with request.app.state.recommender_lock:
                    st["last_error"] = str(e)
                jobs[current_job_id]["status"] = "failed"
                jobs[current_job_id]["error"] = str(e)
                jobs[current_job_id]["finished_at"] = time.time()
            finally:
                async with request.app.state.recommender_lock:
                    st.update({
                        "building": False,
                        "finished_at": time.time(),
                        "current_job_id": None,
                    })

        request.app.state.indexer_task = asyncio.create_task(build_and_swap(job_id))

    return {"status": True, "message": "Reindex scheduled", "job_id": job_id}


@router.get("/reindex/status")
async def _list_reindex_jobs(request: Request):
    jobs = getattr(request.app.state, "reindex_jobs", {})
    payload = sorted(jobs.values(), key=lambda j: j.get("created_at") or 0, reverse=True)
    return {"status": True, "jobs": payload}


@router.get("/reindex/status/{job_id}")
async def _reindex_job_detail(job_id: str, request: Request):
    jobs = getattr(request.app.state, "reindex_jobs", {})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {"status": True, "job": job}


@router.get("/query", response_model=QueryResponse)
def get_query(
    q: str = Query(..., min_length=1, description="User-typed query for tag suggestions"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=100, description="Number of suggestions to return"),
    recommender: RecommenderLogic = Depends(get_recommender)
):
    try:
        suggestions = recommender.suggest(query=q, top_k=k) if len(q.strip()) >= 2 else []
        return {"status": True, "q": q, "suggestions": suggestions}
    except Exception as e:
        print(f"Error in /query endpoint: {e}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


async def _map_hashes_to_db_ids(hashes: List[str]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    if not hashes:
        return out

    CHUNK = 900
    for i in range(0, len(hashes), CHUNK):
        chunk = hashes[i : i + CHUNK]
        q = (
            "SELECT id, funding_institute_id, LOWER(HEX(prof_hash)) AS prof_hash "
            "FROM funding_professors "
            f"WHERE prof_hash IN ({', '.join(['UNHEX(%s)'] * len(chunk))})"
        )
        rows = await DB.fetch_all(q, chunk)
        for r in rows:
            out[str(r["prof_hash"]).lower()] = [r["funding_institute_id"], r["id"]]
    return out


@router.post("/recommend", response_model=RecommendResponse)
async def _post_recommend(
    request: RecommendRequest,
    recommender: RecommenderLogic = Depends(get_recommender),
):
    try:
        ranked = recommender.rank_all(
            tags=request.tags,
            institute_name=getattr(request, "institute_name", None),
            country=getattr(request, "country", None),
            professor_name=getattr(request, "professor_name", None),
            expand_related=request.expand_related,
            domain_filters=getattr(request, "domains", None),
            subfield_filters=getattr(request, "subfields", None),
        )

        all_hashes: List[str] = [str(x).lower() for x in ranked.get("professor_ids", [])]
        explanations = ranked.get("explanations", [])

        if not all_hashes:
            return {"status": True, "professor_ids": [], "explanations": explanations}

        # Apply department filter first if needed (before pagination)
        dept_ids = list(set(getattr(request, "department_ids", []) or []))
        if dept_ids:
            hash_to_dbid = await _map_hashes_to_db_ids(all_hashes)
            filtered_hashes = [h for h in all_hashes if hash_to_dbid.get(h, [-1])[0] in dept_ids]
            
            explanations.append({
                "filter": "department_ids",
                "value": sorted(list(dept_ids)),
                "pre_filter_candidates": len(all_hashes),
                "post_filter_candidates": len(filtered_hashes)
            })
        else:
            filtered_hashes = all_hashes
            hash_to_dbid = {}

        # Apply pagination
        page_size = getattr(request, "page_size", 10) or 10
        page_number = getattr(request, "page_number", 1) or 1
        page_size = max(1, min(200, int(page_size)))
        page_number = max(1, int(page_number))
        start = (page_number - 1) * page_size
        end = start + page_size
        paginated_hashes = filtered_hashes[start:end]

        if not dept_ids:
            hash_to_dbid = await _map_hashes_to_db_ids(paginated_hashes)
        
        professor_ids = [hash_to_dbid[h][1] for h in paginated_hashes if h in hash_to_dbid]

        return {
            "status": True,
            "professor_ids": professor_ids,
            "explanations": explanations,
        }

    except ValueError as e:
        print(f"ValueError in /recommend endpoint: {e}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=400, detail=str(e))
