from __future__ import annotations

import asyncio
import json
import os

from llm_client import (
    Message,
    OpenAIProvider,
    StructuredOutputConfig,
    extract_structured,
    load_env,
)

load_env()


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider = OpenAIProvider(model=model_name)
    try:
        config = StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["unknown", "minor", "major", "critical"],
                    },
                    "owner": {"type": "string"},
                    "risk_summary": {"type": "string"},
                },
                "required": ["priority", "owner", "risk_summary"],
                "additionalProperties": False,
            },
            max_repair_attempts=1,
        )
        result = await extract_structured(
            provider,
            [
                Message.system("Return valid JSON only."),
                Message.user(
                    "Read this incident note and return priority, owner, and risk_summary. "
                    "Incident note: The release is blocked by missing observability dashboards. "
                    "Owner: platform team."
                ),
            ],
            config,
            reasoning_effort="minimal",
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

        print("\n=== Structured Extraction ===\n")
        print(
            json.dumps(
                {
                    "provider": "openai",
                    "model": model_name,
                    "valid": result.valid,
                    "data": result.data,
                    "repair_attempts": result.repair_attempts,
                    "response_kind": result.response_kind,
                    "validation_errors": result.validation_errors,
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
