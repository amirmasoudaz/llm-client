#!/usr/bin/env python3
"""
Push the canonical tag lists from Qdrant (funding_tags collection) back into the
`categories` column of `funding_professors`.

This replaces the legacy CSV-based add_categories script and works with the new
vector-store powered pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from src.config import settings
from collections import defaultdict
from typing import Dict, List, Tuple
from src.db.session import DB
from src.recommender.vector_store import TagVectorStore

def _collect_professor_tags(collection: str, topk: int) -> Dict[str, List[str]]:
    store = TagVectorStore(collection=collection)
    records = store.fetch_all(with_vectors=False)
    prof_to_tags: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for record in records:
        payload = record.payload or {}
        tag = payload.get("tag")
        if not tag:
            continue
        for prof in payload.get("professors") or []:
            pid = str(prof.get("id") or "").lower()
            try:
                weight = float(prof.get("weight", 1.0))
            except Exception:
                weight = 1.0
            if pid:
                prof_to_tags[pid].append((tag.title(), weight))

    trimmed: Dict[str, List[str]] = {}
    for pid, items in prof_to_tags.items():
        ordered = sorted(items, key=lambda x: (-x[1], x[0]))
        deduped: List[str] = []
        seen = set()
        for tag, _ in ordered:
            if tag not in seen:
                deduped.append(tag)
                seen.add(tag)
            if len(deduped) >= topk:
                break
        trimmed[pid] = deduped
    return trimmed


async def _fetch_prof_rows() -> List[dict]:
    sql = """
        SELECT id, LOWER(HEX(prof_hash)) AS prof_hash, categories
        FROM funding_professors
        WHERE prof_hash IS NOT NULL
    """
    return await DB.fetch_all(sql)


async def _apply_updates(updates: List[Tuple[str, int]], batch_size: int = 200, dry_run: bool = False) -> int:
    if dry_run or not updates:
        return len(updates)

    sql = "UPDATE funding_professors SET categories=%s WHERE id=%s"
    for i in range(0, len(updates), batch_size):
        chunk = updates[i : i + batch_size]
        for categories_json, prof_id in chunk:
            await DB.execute(sql, (categories_json, prof_id))
    return len(updates)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sync professor categories from Qdrant tags.")
    parser.add_argument(
        "--collection",
        default=settings.FUNDING_TAG_COLLECTION,
        help="Qdrant collection name containing tag payloads.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=settings.TAG_EXPORT_TOPK,
        help="Maximum number of tags to store per professor.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report intended updates without writing to the database.",
    )
    args = parser.parse_args()

    prof_tags = _collect_professor_tags(args.collection, args.topk)
    if not prof_tags:
        print("No tags found in Qdrant collection; aborting.")
        return

    rows = await _fetch_prof_rows()
    updates: List[Tuple[str, int]] = []
    seen = set()
    for row in rows:
        pid_hex = str(row.get("prof_hash") or "").lower()
        if not pid_hex:
            continue
        if pid_hex not in prof_tags:
            continue
        new_tags = prof_tags[pid_hex]
        new_json = json.dumps(new_tags, ensure_ascii=False)
        updates.append((new_json, row["id"]))

    count = await _apply_updates(updates, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] Would update {count} professors.")
    else:
        print(f"Updated categories for {count} professors.")

    await DB.close()


if __name__ == "__main__":
    asyncio.run(main())
