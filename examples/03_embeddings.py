from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine


async def main() -> None:
    handle = build_live_provider(capability="embeddings")
    try:
        engine = ExecutionEngine(provider=handle.provider)
        result = await engine.embed(
            [
                "robotics safety research",
                "tool calling for autonomous agents",
                "observability for llm infrastructure",
            ]
        )

        print_heading("Embeddings")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "status": result.status,
                "embedding_count": len(result.embeddings),
                "embedding_dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "first_embedding_preview": result.embeddings[0][:8] if result.embeddings else [],
                "usage": summarize_usage(result.usage),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
