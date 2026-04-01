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


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_OPENAI_FILES_MODEL", "gpt-5-mini") or "gpt-5-mini"
    upload_path = example_env("LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH")
    if not upload_path:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_UPLOAD_FILE_PATH to run the OpenAI Files API example.")

    source_path = Path(upload_path)
    if not source_path.exists():
        fail_or_skip(f"Upload path does not exist: {source_path}")

    purpose = example_env("LLM_CLIENT_EXAMPLE_FILE_PURPOSE", "assistants") or "assistants"
    keep_uploaded_file = example_env("LLM_CLIENT_EXAMPLE_KEEP_UPLOADED_FILE", "0") == "1"
    handle = build_provider_handle("openai", model_name)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        with source_path.open("rb") as fh:
            created = await engine.create_file(
                provider_name="openai",
                model=handle.model,
                file=fh,
                purpose=purpose,
            )
        retrieved = await engine.retrieve_file(
            created.file_id,
            provider_name="openai",
            model=handle.model,
        )
        files_page = await engine.list_files(
            provider_name="openai",
            model=handle.model,
            purpose=purpose,
            limit=20,
        )
        downloadable_purposes = {"user_data", "batch", "fine-tune", "vision"}
        content_result = None
        content_unavailable_reason = None
        if purpose in downloadable_purposes:
            content_result = await engine.get_file_content(
                created.file_id,
                provider_name="openai",
                model=handle.model,
            )
        else:
            content_unavailable_reason = (
                f"OpenAI does not allow file content download for purpose={purpose!r}."
            )

        deletion = None
        if not keep_uploaded_file:
            deletion = await engine.delete_file(
                created.file_id,
                provider_name="openai",
                model=handle.model,
            )

        print_heading("OpenAI Files API")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "purpose": purpose,
                "source_path": str(source_path),
                "created": created.to_dict(),
                "retrieved": retrieved.to_dict(),
                "listed_file_ids": [item.file_id for item in files_page.items],
                "content": (
                    {
                        "file_id": content_result.file_id,
                        "media_type": content_result.media_type,
                        "byte_length": content_result.byte_length,
                        "preview_text": content_result.content[:200].decode("utf-8", errors="replace"),
                    }
                    if content_result is not None
                    else None
                ),
                "content_unavailable_reason": content_unavailable_reason,
                "deleted": deletion.to_dict() if deletion is not None else None,
                "kept_uploaded_file": keep_uploaded_file,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
