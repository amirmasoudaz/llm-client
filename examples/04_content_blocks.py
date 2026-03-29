from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.content import (
    AudioBlock,
    ContentHandlingMode,
    ContentMessage,
    ContentRequestEnvelope,
    FileBlock,
    MetadataBlock,
    TextBlock,
    ensure_completion_result,
    project_content_blocks,
)
from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Role


async def main() -> None:
    handle = build_live_provider()
    try:
        engine = ExecutionEngine(provider=handle.provider)
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
            provider=handle.name,
            mode=ContentHandlingMode.LOSSY,
            include_metadata=True
        )
        projection_matrix = {
            provider_name: {
                "projected_blocks": [block.to_dict() for block in project_content_blocks(
                    source_blocks,
                    provider=provider_name,
                    mode=ContentHandlingMode.LOSSY,
                ).blocks],
                "degradations": [
                    item.reason
                    for item in project_content_blocks(
                        source_blocks,
                        provider=provider_name,
                        mode=ContentHandlingMode.LOSSY,
                    ).degradations
                ],
            }
            for provider_name in ("openai", "anthropic", "google")
        }

        request_envelope = ContentRequestEnvelope(
            provider=handle.name,
            model=handle.model,
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

        print_heading("Source Content Bundle")
        print_json(
            {
                "source_blocks": [block.to_dict() for block in source_blocks],
            }
        )

        print_heading("Provider Projection Matrix")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "active_provider_projection": {
                    "projected_block_count": len(provider_projection.blocks),
                    "projected_blocks": [block.to_dict() for block in provider_projection.blocks],
                    "degradations": [item.reason for item in provider_projection.degradations],
                },
                "other_provider_views": projection_matrix,
            }
        )

        print_heading("Content Envelopes + Real Completion")
        print_json(
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
                "usage": summarize_usage(result.usage),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
