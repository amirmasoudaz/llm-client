from __future__ import annotations

import asyncio
import json
import os

from llm_client import ExecutionEngine, OpenAIProvider, Role, load_env
from llm_client.content import (
    AudioBlock,
    FileBlock,
    MetadataBlock,
    TextBlock,
    ContentMessage,
    ContentHandlingMode,
    ContentRequestEnvelope,
    ensure_completion_result,
    project_content_blocks,
)

load_env()


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=4, ensure_ascii=False, default=str))


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider_name = "openai"
    provider = OpenAIProvider(model=model_name)
    try:
        engine = ExecutionEngine(provider=provider)
        source_blocks = [
            TextBlock(
                "You are reading a release handoff bundle. Produce a release brief with: "
                "overall status, key risks, and recommended next step."
            ),
            TextBlock(
                "Core notes: release readiness is high. Remaining work is extraction cleanup and a final consumer migration pass."
            ),
            AudioBlock(
                transcript=(
                    "Stakeholder audio transcript: leadership is comfortable shipping after dashboards and "
                    "consumer migration notes are confirmed."
                )
            ),
            FileBlock(
                name="release_readiness_review.pdf",
                mime_type="application/pdf",
            ),
            MetadataBlock(
                {
                    "ticket": "REL-204",
                    "owner": "platform-foundations",
                    "severity": "medium",
                }
            ),
        ]

        provider_projection = project_content_blocks(
            source_blocks,
            provider=provider_name,
            mode=ContentHandlingMode.LOSSY,
            include_metadata=True,
        )
        projection_matrix = {
            name: {
                "projected_blocks": [block.to_dict() for block in project_content_blocks(
                    source_blocks,
                    provider=name,
                    mode=ContentHandlingMode.LOSSY,
                ).blocks],
                "degradations": [
                    item.reason
                    for item in project_content_blocks(
                        source_blocks,
                        provider=name,
                        mode=ContentHandlingMode.LOSSY,
                    ).degradations
                ],
            }
            for name in ("openai", "anthropic", "google")
        }

        request_envelope = ContentRequestEnvelope(
            provider=provider_name,
            model=model_name,
            messages=(
                ContentMessage(
                    role=Role.SYSTEM,
                    blocks=(
                        TextBlock(
                            "You are a release-operations analyst. Read the provided bundle carefully, "
                            "treat metadata as operational context, and respond with a concise release brief."
                        ),
                    ),
                ),
                ContentMessage(
                    role=Role.USER,
                    blocks=provider_projection.blocks,
                ),
            ),
        )

        response = await engine.complete_content(request_envelope)
        result = ensure_completion_result(response)

        usage = (
            {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "total_cost": result.usage.total_cost,
            }
            if result.usage is not None
            else {}
        )

        print("\n=== Source Content Bundle ===\n")
        _print_json(
            {
                "source_blocks": [block.to_dict() for block in source_blocks],
            }
        )

        print("\n=== Provider Projection Matrix ===\n")
        _print_json(
            {
                "provider": provider_name,
                "model": model_name,
                "active_provider_projection": {
                    "projected_block_count": len(provider_projection.blocks),
                    "projected_blocks": [block.to_dict() for block in provider_projection.blocks],
                    "degradations": [item.reason for item in provider_projection.degradations],
                },
                "other_provider_views": projection_matrix,
            }
        )

        print("\n=== Content Envelopes + Real Completion ===\n")
        _print_json(
            {
                "request_messages_sent": [
                    {
                        "role": message.role.value,
                        "blocks": [block.to_dict() for block in message.blocks],
                    }
                    for message in request_envelope.messages
                ],
                "response_blocks": [block.to_dict() for block in response.message.blocks],
                "text": result.content,
                "status": result.status,
                "usage": usage,
            }
        )
        print(f"\n\n=== Final Brief ===\n\n{result.content}\n")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
