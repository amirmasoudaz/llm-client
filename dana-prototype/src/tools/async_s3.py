from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, BinaryIO, Optional, Iterable, Any

import aiofiles
from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ConnectionClosedError,
    ReadTimeoutError,
)


# -----------------------------
# Rate limiting primitives
# -----------------------------

class AsyncTokenBucket:
    """
    Simple token bucket rate limiter for async code.
    Rate is tokens per second; capacity is max burst.
    Call await acquire(tokens=1) before a request.
    """
    def __init__(self, rate: float, capacity: float):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be > 0")
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return

        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now

                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                missing = tokens - self._tokens
                wait_time = missing / self._rate

            await asyncio.sleep(wait_time)


class AsyncConcurrencyLimiter:
    """
    Simple semaphore-based concurrency limiter.
    """
    def __init__(self, max_concurrency: int):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        self._sem = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()
        return False


# -----------------------------
# Retry / backoff
# -----------------------------

def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            EndpointConnectionError,
            ConnectionClosedError,
            ReadTimeoutError,
            asyncio.TimeoutError,
        ),
    )


def _is_retryable_client_error(e: ClientError) -> bool:
    # Retry common transient S3 errors
    code = (e.response or {}).get("Error", {}).get("Code", "")
    status = (e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0)

    retryable_codes = {
        "RequestTimeout",
        "Throttling",
        "ThrottlingException",
        "SlowDown",
        "InternalError",
        "ServiceUnavailable",
        "TransientError",
    }
    if code in retryable_codes:
        return True

    # 5xx and some 429
    if status in {429, 500, 502, 503, 504}:
        return True

    return False


async def _retry(
    fn,
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    jitter: float,
):
    last_exc = None
    for i in range(attempts):
        try:
            return await fn()
        except ClientError as e:
            last_exc = e
            if not _is_retryable_client_error(e) or i == attempts - 1:
                raise
        except BaseException as e:
            last_exc = e
            if not _is_retryable_exception(e) or i == attempts - 1:
                raise

        delay = min(max_delay, base_delay * (2 ** i))
        delay = delay + random.random() * jitter
        await asyncio.sleep(delay)

    raise last_exc  # should be unreachable


# -----------------------------
# Settings
# -----------------------------

@dataclass(frozen=True)
class S3ClientSettings:
    region_name: str
    bucket: str

    # Connection pool settings
    max_pool_connections: int = 64

    # Concurrency limiter around all S3 calls from this wrapper
    max_concurrency: int = 64

    # Rate limiting: requests per second, burst capacity
    # Set to None to disable
    reqs_per_sec: Optional[float] = 50.0
    req_burst: Optional[float] = 100.0

    # Timeouts
    connect_timeout: int = 10
    read_timeout: int = 60

    # Retries
    retry_attempts: int = 8
    retry_base_delay: float = 0.25
    retry_max_delay: float = 8.0
    retry_jitter: float = 0.25

    # Multipart upload defaults
    multipart_threshold_bytes: int = 100 * 1024 * 1024  # 100 MB
    part_size_bytes: int = 16 * 1024 * 1024            # 16 MB, must be >= 5MB

    # Integrity
    compute_sha256: bool = False  # if True, compute sha256 for local uploads


# -----------------------------
# Async S3 wrapper
# -----------------------------

class AsyncS3:
    """
    Production-grade async S3 helper:
    - aiobotocore-based, true async
    - connection pooling via botocore Config
    - concurrency limiter + token-bucket rate limiter
    - robust retries with exponential backoff + jitter
    - common operations: put/get/delete/list, prefix delete
    - file uploads/downloads with optional multipart upload
    - streaming download iterator
    """

    def __init__(self, settings: S3ClientSettings):
        self.settings = settings

        self._session = get_session()
        self._client_cm = None
        self._client = None

        self._concurrency = AsyncConcurrencyLimiter(settings.max_concurrency)
        self._rate = None
        if settings.reqs_per_sec and settings.req_burst:
            self._rate = AsyncTokenBucket(settings.reqs_per_sec, settings.req_burst)

        self._cfg = Config(
            region_name=settings.region_name,
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
            retries={"max_attempts": 0, "mode": "standard"},  # we do our own retries
            max_pool_connections=settings.max_pool_connections,
        )

    async def __aenter__(self) -> "AsyncS3":
        # Credentials are resolved by botocore's standard chain:
        # env vars, ~/.aws, instance role, ECS task role, etc.
        self._client_cm = self._session.create_client("s3", config=self._cfg)
        self._client = await self._client_cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client_cm:
            await self._client_cm.__aexit__(exc_type, exc, tb)
        self._client_cm = None
        self._client = None

    def _require_client(self):
        if not self._client:
            raise RuntimeError("AsyncS3 must be used inside 'async with AsyncS3(...)'")

    async def _guard(self):
        if self._rate:
            await self._rate.acquire(1.0)
        return self._concurrency

    async def _call(self, coro_factory):
        self._require_client()

        async def wrapped():
            async with (await self._guard()):
                return await coro_factory()

        return await _retry(
            wrapped,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
            jitter=self.settings.retry_jitter,
        )

    # -------------------------
    # Key helpers
    # -------------------------

    @staticmethod
    def join_key(*parts: str) -> str:
        cleaned = []
        for p in parts:
            if p is None:
                continue
            p = str(p).replace("\\", "/").strip("/")
            if p:
                cleaned.append(p)
        return "/".join(cleaned)

    # -------------------------
    # Basic ops
    # -------------------------

    async def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        async def do():
            return await self._client.put_object(
                Bucket=self.settings.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )

        await self._call(do)

    async def get_bytes(self, key: str) -> bytes:
        async def do():
            return await self._client.get_object(Bucket=self.settings.bucket, Key=key)

        resp = await self._call(do)
        body = resp["Body"]
        try:
            return await body.read()
        finally:
            body.close()

    async def head(self, key: str) -> dict[str, Any]:
        async def do():
            return await self._client.head_object(Bucket=self.settings.bucket, Key=key)
        return await self._call(do)

    async def exists(self, key: str) -> bool:
        try:
            await self.head(key)
            return True
        except ClientError as e:
            status = (e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if status == 404:
                return False
            raise

    async def delete(self, key: str) -> None:
        async def do():
            return await self._client.delete_object(Bucket=self.settings.bucket, Key=key)
        await self._call(do)

    async def list_keys(
        self,
        prefix: str,
        *,
        limit: Optional[int] = None,
    ) -> list[str]:
        self._require_client()

        out: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")

        async def page_iter():
            async for page in paginator.paginate(
                Bucket=self.settings.bucket,
                Prefix=prefix,
            ):
                yield page

        async for page in page_iter():
            for obj in page.get("Contents", []):
                out.append(obj["Key"])
                if limit is not None and len(out) >= limit:
                    return out
        return out

    async def delete_prefix(self, prefix: str) -> int:
        """
        Deletes all objects under a prefix. Uses batch delete (1000 per request).
        Returns number deleted (best-effort count).
        """
        self._require_client()

        paginator = self._client.get_paginator("list_objects_v2")
        deleted = 0
        batch: list[dict[str, str]] = []

        async for page in paginator.paginate(Bucket=self.settings.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                batch.append({"Key": obj["Key"]})
                if len(batch) == 1000:
                    await self._delete_batch(batch)
                    deleted += len(batch)
                    batch = []

        if batch:
            await self._delete_batch(batch)
            deleted += len(batch)

        return deleted

    async def _delete_batch(self, objects: list[dict[str, str]]) -> None:
        async def do():
            return await self._client.delete_objects(
                Bucket=self.settings.bucket,
                Delete={"Objects": objects, "Quiet": True},
            )
        await self._call(do)

    # -------------------------
    # File upload/download
    # -------------------------

    async def upload_file(
        self,
        local_path: str,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Upload local file to S3.
        If file >= multipart_threshold_bytes, uses multipart upload.
        """
        size = os.path.getsize(local_path)
        if size >= self.settings.multipart_threshold_bytes:
            await self._multipart_upload(local_path, key, content_type=content_type, metadata=metadata)
            return

        sha256 = None
        if self.settings.compute_sha256:
            sha256 = await _sha256_file_async(local_path)

        async with aiofiles.open(local_path, "rb") as f:
            data = await f.read()

        # You can store sha256 as metadata if you want integrity checks later
        md = dict(metadata or {})
        if sha256:
            md["sha256"] = sha256

        await self.put_bytes(key, data, content_type=content_type, metadata=md)

    async def download_file(self, key: str, local_path: str) -> None:
        """
        Download S3 object to local path (streamed).
        """
        async def do():
            return await self._client.get_object(Bucket=self.settings.bucket, Key=key)

        resp = await self._call(do)
        body = resp["Body"]

        try:
            async with aiofiles.open(local_path, "wb") as f:
                while True:
                    chunk = await body.read(1024 * 1024)
                    if not chunk:
                        break
                    await f.write(chunk)
        finally:
            body.close()

    async def iter_download(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        """
        Async generator streaming download in chunks.
        Useful for FastAPI StreamingResponse.
        """
        async def do():
            return await self._client.get_object(Bucket=self.settings.bucket, Key=key)

        resp = await self._call(do)
        body = resp["Body"]

        try:
            while True:
                chunk = await body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    # -------------------------
    # Multipart upload
    # -------------------------

    async def _multipart_upload(
        self,
        local_path: str,
        key: str,
        *,
        content_type: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        part_size = max(self.settings.part_size_bytes, 5 * 1024 * 1024)
        file_size = os.path.getsize(local_path)

        async def create():
            return await self._client.create_multipart_upload(
                Bucket=self.settings.bucket,
                Key=key,
                ContentType=content_type,
                Metadata=metadata or {},
            )

        resp = await self._call(create)
        upload_id = resp["UploadId"]

        parts: list[dict[str, Any]] = []
        part_number = 1

        try:
            async with aiofiles.open(local_path, "rb") as f:
                offset = 0
                while offset < file_size:
                    data = await f.read(part_size)
                    if not data:
                        break

                    pn = part_number
                    part_number += 1
                    offset += len(data)

                    etag = await self._upload_part(key, upload_id, pn, data)
                    parts.append({"ETag": etag, "PartNumber": pn})

            async def complete():
                return await self._client.complete_multipart_upload(
                    Bucket=self.settings.bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            await self._call(complete)

        except BaseException:
            async def abort():
                return await self._client.abort_multipart_upload(
                    Bucket=self.settings.bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            try:
                await self._call(abort)
            except BaseException:
                pass
            raise

    async def _upload_part(self, key: str, upload_id: str, part_number: int, data: bytes) -> str:
        async def do():
            return await self._client.upload_part(
                Bucket=self.settings.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data,
            )
        resp = await self._call(do)
        return resp["ETag"]


async def _sha256_file_async(path: str) -> str:
    h = hashlib.sha256()
    async with aiofiles.open(path, "rb") as f:
        while True:
            chunk = await f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    async def main():
        settings = S3ClientSettings(
            region_name="ca-central-1",
            bucket="canapply-platform-prod",
            max_pool_connections=64,
            max_concurrency=64,
            reqs_per_sec=80.0,
            req_burst=160.0,
        )

        async with AsyncS3(settings) as s3:
            key = s3.join_key("platform", "funding_attachments", "test.txt")
            await s3.put_bytes(key, b"hello", content_type="text/plain")
            
            data = await s3.get_bytes(key)
            print(data)

            # key = s3.join_key("platform", "funding_attachments", "student_123", "doc.pdf")
            # await s3.upload_file("/tmp/doc.pdf", key, content_type="application/pdf")

            await s3.delete(key)

    asyncio.run(main())

    """
    from fastapi import FastAPI
    from starlette.responses import StreamingResponse

    app = FastAPI()

    @app.get("/files/{key_path:path}")
    async def download(key_path: str):
        async with AsyncS3(settings) as s3:
            key = s3.join_key("platform", key_path)
            return StreamingResponse(
                s3.iter_download(key),
                media_type="application/octet-stream",
            )
    """