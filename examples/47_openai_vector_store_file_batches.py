from __future__ import annotations

import asyncio
from pathlib import Path

from cookbook_support import (
    build_live_provider,
    close_provider,
    example_env,
    fail_or_skip,
    print_heading,
    print_json,
)

from llm_client.engine import ExecutionEngine


def _csv_env(name: str) -> list[str]:
    raw = example_env(name, "") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


async def main() -> None:
    handle = build_live_provider()
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI vector-store-file-batches example.")

    vector_store_id = example_env("LLM_CLIENT_EXAMPLE_VECTOR_STORE_ID")
    if not vector_store_id:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_VECTOR_STORE_ID to run the vector-store file batch example.")

    file_ids = _csv_env("LLM_CLIENT_EXAMPLE_VECTOR_STORE_FILE_IDS")
    upload_paths = _csv_env("LLM_CLIENT_EXAMPLE_VECTOR_STORE_UPLOAD_PATHS")
    if not file_ids and not upload_paths:
        fail_or_skip(
            "Set LLM_CLIENT_EXAMPLE_VECTOR_STORE_FILE_IDS or "
            "LLM_CLIENT_EXAMPLE_VECTOR_STORE_UPLOAD_PATHS to run the vector-store file batch example."
        )

    missing_paths = [path for path in upload_paths if not Path(path).exists()]
    if missing_paths:
        fail_or_skip(f"Vector-store upload paths do not exist: {missing_paths}")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        if file_ids:
            batch = await engine.create_vector_store_file_batch_and_poll(
                vector_store_id,
                provider_name="openai",
                model=handle.model,
                file_ids=file_ids,
            )
        else:
            batch = await engine.upload_vector_store_file_batch_and_poll(
                vector_store_id,
                provider_name="openai",
                model=handle.model,
                files=upload_paths,
            )
        retrieved = await engine.retrieve_vector_store_file_batch(
            vector_store_id,
            batch.batch_id,
            provider_name="openai",
            model=handle.model,
        )
        batch_files = await engine.list_vector_store_file_batch_files(
            vector_store_id,
            batch.batch_id,
            provider_name="openai",
            model=handle.model,
            limit=100,
        )

        print_heading("OpenAI Vector Store File Batches")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "vector_store_id": vector_store_id,
                "batch": batch.to_dict(),
                "retrieved_batch": retrieved.to_dict(),
                "batch_file_ids": [item.file_id for item in batch_files.items],
                "batch_files": [item.to_dict() for item in batch_files.items],
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
