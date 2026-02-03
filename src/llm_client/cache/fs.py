"""
FileSystem cache backend.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..concurrency import run_sync
from .base import CacheBackendName


@dataclass
class FSCacheConfig:
    dir: Path
    client_type: str
    default_collection: str = "default"
    name: CacheBackendName = "fs"


class FSCache:
    name: CacheBackendName = "fs"

    def __init__(self, cfg: FSCacheConfig) -> None:
        self.cfg = cfg
        self.default_collection = cfg.default_collection
        self.client_type = cfg.client_type
        self.cfg.dir.mkdir(parents=True, exist_ok=True)

    def _get_collection(self, collection: str | None) -> str:
        return collection or self.default_collection

    async def ensure_ready(self) -> None:
        return

    async def close(self) -> None:
        return

    async def warm(self) -> None:
        return

    def _path_for(self, key: str, collection: str | None = None) -> Path:
        coll = self._get_collection(collection)
        coll_dir = self.cfg.dir / coll
        coll_dir.mkdir(parents=True, exist_ok=True)
        return coll_dir / f"{key}.json"

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return self._path_for(effective_key, collection).exists()

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        if rewrite_cache and not regen_cache and self.client_type == "completions":
            for i in range(0, 1000):
                cand = f"{identifier}_{i}"
                if not await self.exists(cand, collection):
                    return cand, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str, collection: str | None = None) -> dict | None:
        path = self._path_for(effective_key, collection)
        try:
            raw = await run_sync(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def write(
        self,
        effective_key: str,
        response: dict,
        model_name: str,
        collection: str | None = None,
    ) -> None:
        path = self._path_for(effective_key, collection)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(response, indent=2, ensure_ascii=False)

        def _write_atomic() -> None:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)

        # Atomic replace to avoid partial writes/corruption.
        await run_sync(_write_atomic)
