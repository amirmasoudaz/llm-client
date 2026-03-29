from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.content import (
    ContentHandlingMode,
    ContentMessage,
    ContentRequestEnvelope,
    FileBlock,
    Role,
    TextBlock,
    content_blocks_to_openai_chat_content,
    content_blocks_to_openai_responses_content,
    prepare_file_block,
    project_content_blocks,
)
from llm_client.engine import ExecutionEngine
from llm_client.providers import OpenAIProvider


DEMO_FILE_TEXT = """Release mission control memo

Overall status: nearly ready to ship.
Remaining work: finalize FileBlock semantics, complete package guides, and run release candidate validation.
Risks: residual drift between examples and package docs, plus any last-minute provider-specific transport mismatch.
Recommended next step: cut RC1 after final package docs and artifact verification pass.
"""


async def main() -> None:
    handle = build_live_provider()
    temp_path = Path(tempfile.gettempdir()) / "llm_client_demo_release_bundle.txt"
    temp_path.write_text(DEMO_FILE_TEXT, encoding="utf-8")

    source_file = FileBlock(
        file_path=str(temp_path),
        name="release_bundle.txt",
        mime_type="text/plain",
        extracted_text=DEMO_FILE_TEXT,
    )
    prepared = prepare_file_block(source_file)

    projection_matrix = {
        "openai_chat": content_blocks_to_openai_chat_content([prepared], mode=ContentHandlingMode.LOSSY),
        "openai_responses": content_blocks_to_openai_responses_content([prepared], mode=ContentHandlingMode.LOSSY),
        "anthropic": [block.to_dict() for block in project_content_blocks([prepared], provider="anthropic").blocks],
        "google": [block.to_dict() for block in project_content_blocks([prepared], provider="google").blocks],
    }

    native_provider = None
    active_provider_name = handle.name
    active_model = handle.model
    response_payload: dict[str, object]
    try:
        if handle.name == "openai":
            native_provider = OpenAIProvider(model=handle.model, use_responses_api=True)
            engine = ExecutionEngine(provider=native_provider)
            active_provider_name = "openai-responses"
            request = ContentRequestEnvelope(
                provider="openai",
                model=handle.model,
                messages=(
                    ContentMessage(
                        role=Role.USER,
                        blocks=(
                            TextBlock(
                                "Read the attached release bundle and return a concise readiness summary with status, risks, and next step."
                            ),
                            prepared,
                        ),
                    ),
                ),
            )
        else:
            engine = ExecutionEngine(provider=handle.provider)
            fallback_blocks = project_content_blocks(
                [
                    TextBlock(
                        "Read the attached release bundle and return a concise readiness summary with status, risks, and next step."
                    ),
                    prepared,
                ],
                provider=handle.name,
                mode=ContentHandlingMode.LOSSY,
            ).blocks
            request = ContentRequestEnvelope(
                provider=handle.name,
                model=handle.model,
                messages=(ContentMessage(role=Role.USER, blocks=fallback_blocks),),
            )

        response = await engine.complete_content(request)
        response_payload = {
            "provider": active_provider_name,
            "model": active_model,
            "response_blocks": [block.to_dict() for block in response.message.blocks],
            "status": response.status,
            "error": response.error,
            "usage": summarize_usage(response.usage),
        }
    finally:
        await close_provider(handle.provider)
        if native_provider is not None:
            await close_provider(native_provider)

    print_heading("Canonical FileBlock")
    print_json(
        {
            "source_file": source_file.to_dict(),
            "prepared_file": prepared.to_dict(),
        }
    )

    print_heading("Transport Matrix")
    print_json(projection_matrix)

    print_heading("Real Completion")
    print_json(response_payload)


if __name__ == "__main__":
    asyncio.run(main())
