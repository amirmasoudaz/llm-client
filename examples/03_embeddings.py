from __future__ import annotations

import asyncio
import json
import os

from llm_client import ExecutionEngine, OpenAIProvider, load_env

load_env()


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL", "text-embedding-3-small")
    provider = OpenAIProvider(model=model_name)
    try:
        engine = ExecutionEngine(provider=provider)
        result = await engine.embed(
            inputs=[
                "robotics safety research",
                "tool calling for autonomous agents",
                "observability for llm infrastructure",
            ]
        )

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

        print("\n=== Embeddings ===\n")
        print(
            json.dumps(
                {
                    "provider": "openai",
                    "model": model_name,
                    "status": result.status,
                    "embedding_count": len(result.embeddings),
                    "embedding_dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                    "first_embedding_preview": result.embeddings[0][:8] if result.embeddings else [],
                    "usage": usage,
                },
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
