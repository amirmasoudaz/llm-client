#!/usr/bin/env python3
import argparse
import asyncio
import json
from src.config import settings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import aiofiles

# Try fastest JSON first
try:
    import orjson as _json  # pip install orjson
    def jloads(s: bytes | str):
        return _json.loads(s)
except Exception:
    try:
        import ujson as _json
        def jloads(s: str):
            return _json.loads(s)
    except Exception:
        def jloads(s: str):
            return json.loads(s)

try:
    from blake3 import blake3
    def u64_hash(s: str) -> int:
        return int.from_bytes(blake3(s.encode("utf-8")).digest()[:8], "big", signed=False)
except Exception:
    import hashlib
    def u64_hash(s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big", signed=False)


def qdrant_headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["api-key"] = api_key
    return h

async def ensure_collection(session: aiohttp.ClientSession, base_url: str, collection: str, api_key: Optional[str]) -> None:
    # Check exists
    url_get = f"{base_url}/collections/{collection}"
    async with session.get(url_get, headers=qdrant_headers(api_key)) as r:
        if r.status == 200:
            return
    # Create a simple 1D vector collection; store cache in payload
    url_put = f"{base_url}/collections/{collection}"
    body = {
        "vectors": {"size": 1, "distance": "Dot"},
        "on_disk_payload": True,
        "optimizers_config": {"default_segment_number": 2},
    }
    async with session.put(url_put, headers=qdrant_headers(api_key), data=json.dumps(body)) as r:
        if r.status not in (200, 201):
            txt = await r.text()
            raise RuntimeError(f"Failed to create collection '{collection}': {r.status} {txt}")

async def upsert_points(session: aiohttp.ClientSession, base_url: str, collection: str, points: List[Dict], api_key: Optional[str], retries: int = 5) -> None:
    if not points:
        return
    url = f"{base_url}/collections/{collection}/points?wait=true"
    body = {"points": points}
    payload = json.dumps(body)
    delay = 0.5
    for attempt in range(retries):
        try:
            async with session.put(url, headers=qdrant_headers(api_key), data=payload) as r:
                if r.status in (200, 202):
                    return
                txt = await r.text()
                raise RuntimeError(f"HTTP {r.status}: {txt}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

# -----------------------------
# File → point
# -----------------------------
async def load_point(path: Path, client_type: str) -> Optional[Dict]:
    try:
        async with aiofiles.open(path, "rb") as f:
            raw = await f.read()
        data = jloads(raw)
        if not isinstance(data, dict):
            return None
        if data.get("error") != "OK":
            return None

        effective_identifier = path.stem  # includes suffix like _3 if present
        base_identifier = data.get("identifier", effective_identifier)
        params = data.get("params") or {}
        model_name = params.get("model", "unknown")
        status = data.get("status", None)

        payload = {
            "identifier": effective_identifier,
            "base_identifier": base_identifier,
            "client_type": client_type,       # "embeddings" | "completions"
            "model": model_name,
            "status": status,
            "error": data.get("error"),
            "created_at": int(path.stat().st_mtime),
            "cache": data,                    # full cached response blob
        }

        return {
            "id": u64_hash(effective_identifier),
            "vector": [0.0],                  # dummy 1D vector
            "payload": payload,
        }
    except Exception:
        return None

# -----------------------------
# Migration (concurrent pipeline)
# -----------------------------
async def migrate_dir(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: Optional[str],
    src_dir: Path,
    collection: str,
    client_type: str,
    batch_size: int,
    read_concurrency: int,
    upsert_concurrency: int,
) -> Tuple[int, int]:
    await ensure_collection(session, base_url, collection, api_key)

    files = list(src_dir.glob("*.json"))
    total_files = len(files)
    if total_files == 0:
        return 0, 0

    # Queues
    points_q: asyncio.Queue = asyncio.Queue(maxsize=batch_size * 4)
    done_sentinel = object()

    migrated = 0
    skipped = 0
    migrated_lock = asyncio.Lock()

    async def reader_worker(worker_id: int, paths: List[Path]):
        nonlocal skipped
        for p in paths:
            point = await load_point(p, client_type)
            if point is None:
                skipped += 1
            else:
                await points_q.put(point)
        # readers signal by putting sentinel once all finished (coordinated outside)

    async def batch_sender():
        nonlocal migrated
        # Semaphore to limit concurrent upserts
        sem = asyncio.Semaphore(upsert_concurrency)

        async def send_batch(batch: List[Dict]):
            nonlocal migrated
            async with sem:
                await upsert_points(session, base_url, collection, batch, api_key)
                async with migrated_lock:
                    migrated += len(batch)

        pending: List[asyncio.Task] = []
        current: List[Dict] = []

        active = True
        while active:
            item = await points_q.get()
            if item is done_sentinel:
                # flush remaining
                if current:
                    pending.append(asyncio.create_task(send_batch(current)))
                    current = []
                active = False
                break
            current.append(item)
            if len(current) >= batch_size:
                pending.append(asyncio.create_task(send_batch(current)))
                current = []

        # Wait for in-flight batches
        if pending:
            await asyncio.gather(*pending)

    # Split file list over readers
    if read_concurrency <= 0:
        read_concurrency = 8
    chunk = max(1, len(files) // read_concurrency)
    reader_tasks = []
    for i in range(0, len(files), chunk):
        reader_tasks.append(asyncio.create_task(reader_worker(i // chunk, files[i : i + chunk])))

    sender_task = asyncio.create_task(batch_sender())

    # Progress printer (optional)
    async def progress():
        last = -1
        while any(not t.done() for t in reader_tasks):
            done = migrated + skipped
            if done != last:
                print(f"\rProcessed {done}/{total_files} (migrated={migrated}, skipped={skipped})", end="", flush=True)
                last = done
            await asyncio.sleep(0.5)
        # Final update
        print(f"\rProcessed {migrated + skipped}/{total_files} (migrated={migrated}, skipped={skipped})")

    prog_task = asyncio.create_task(progress())

    # Wait for readers
    await asyncio.gather(*reader_tasks)
    # Signal sender to flush & stop
    await points_q.put(done_sentinel)
    await sender_task
    await prog_task

    return migrated, skipped


async def main():
    parser = argparse.ArgumentParser(description="Fast migrate local JSON caches into Qdrant (payload cache).")
    parser.add_argument("--project-dir", required=True, help="Path to your <project_dir>")
    parser.add_argument("--qdrant", default=settings.QDRANT_URL, help="Qdrant base URL")
    parser.add_argument("--api-key", default=settings.QDRANT_API_KEY, help="Qdrant API key (optional; local usually empty)")
    parser.add_argument("--embeddings-collection", default="embeddings")
    parser.add_argument("--completions-collection", default="tag_generation")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--read-concurrency", type=int, default=16)
    parser.add_argument("--upsert-concurrency", type=int, default=6)
    args = parser.parse_args()

    base_url = args.qdrant.rstrip("/")
    project = Path(args.project_dir).resolve()
    dir_embeddings = project / "cache" / "embeddings"
    dir_completions = project / "cache" / "tag_generation"

    if not dir_embeddings.exists() and not dir_completions.exists():
        raise SystemExit(f"No cache directories found at:\n  {dir_embeddings}\n  {dir_completions}")

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        total_migrated = 0
        total_skipped = 0

        if dir_embeddings.exists():
            print(f"→ Migrating embeddings from {dir_embeddings} → '{args.embeddings_collection}' …")
            m, s = await migrate_dir(
                session,
                base_url,
                args.api_key,
                dir_embeddings,
                args.embeddings_collection,
                client_type="embeddings",
                batch_size=args.batch_size,
                read_concurrency=args.read_concurrency,
                upsert_concurrency=args.upsert_concurrency,
            )
            print(f"Embeddings done. Migrated: {m}, Skipped: {s}")
            total_migrated += m
            total_skipped += s

        if dir_completions.exists():
            print(f"→ Migrating completions from {dir_completions} → '{args.completions_collection}' …")
            m, s = await migrate_dir(
                session,
                base_url,
                args.api_key,
                dir_completions,
                args.completions_collection,
                client_type="completions",
                batch_size=args.batch_size,
                read_concurrency=args.read_concurrency,
                upsert_concurrency=args.upsert_concurrency,
            )
            print(f"Completions done. Migrated: {m}, Skipped: {s}")
            total_migrated += m
            total_skipped += s

        print(f"\nMigration complete. Total migrated: {total_migrated}, total skipped: {total_skipped}")

if __name__ == "__main__":
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main())
