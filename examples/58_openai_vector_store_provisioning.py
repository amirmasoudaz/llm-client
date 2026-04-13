from __future__ import annotations

import asyncio
from pathlib import Path

from cookbook_support import (
    build_provider_handle,
    close_provider,
    example_env,
    fail_or_skip,
    print_heading,
    print_json,
)

from llm_client.engine import ExecutionEngine
from llm_client.tools import ResponsesChunkingStrategy, ResponsesVectorStoreFileSpec


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_OPENAI_RETRIEVAL_MODEL", "gpt-5-mini") or "gpt-5-mini"
    upload_path = example_env("LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH")
    if not upload_path:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH to run the vector-store provisioning example.")

    source_path = Path(upload_path)
    if not source_path.exists():
        fail_or_skip(f"Upload path does not exist: {source_path}")

    keep_uploaded_file = example_env("LLM_CLIENT_EXAMPLE_KEEP_UPLOADED_FILE", "0") == "1"
    keep_vector_store = example_env("LLM_CLIENT_EXAMPLE_KEEP_VECTOR_STORE", "0") == "1"
    handle = build_provider_handle("openai", model_name)

    created_file = None
    vector_store = None

    try:
        engine = ExecutionEngine(provider=handle.provider)
        with source_path.open("rb") as fh:
            created_file = await engine.create_file(
                provider_name="openai",
                model=handle.model,
                file=fh,
                purpose="assistants",
            )

        vector_store = await engine.create_vector_store_and_poll(
            provider_name="openai",
            model=handle.model,
            name="Cookbook Provisioned Store",
            files=[
                ResponsesVectorStoreFileSpec(
                    file_id=created_file.file_id,
                    attributes={"scope": "cookbook", "source": source_path.name},
                    chunking_strategy=ResponsesChunkingStrategy.auto(),
                )
            ],
            poll_interval=0.0,
            timeout=60.0,
        )
        search_result = await engine.search_vector_store(
            vector_store.vector_store_id,
            query="What document was provisioned into this vector store?",
            provider_name="openai",
            model=handle.model,
            max_num_results=3,
        )

        print_heading("OpenAI Vector Store Provisioning")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "source_path": str(source_path),
                "created_file": created_file.to_dict(),
                "vector_store": vector_store.to_dict(),
                "search_result": search_result.to_dict(),
            }
        )
    finally:
        if vector_store is not None and not keep_vector_store:
            try:
                await engine.delete_vector_store(
                    vector_store.vector_store_id,
                    provider_name="openai",
                    model=handle.model,
                )
            except Exception:
                pass
        if created_file is not None and not keep_uploaded_file:
            try:
                await engine.delete_file(
                    created_file.file_id,
                    provider_name="openai",
                    model=handle.model,
                )
            except Exception:
                pass
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
