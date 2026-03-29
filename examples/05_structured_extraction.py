from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, print_json

from llm_client.providers.types import Message
from llm_client.structured import StructuredOutputConfig, extract_structured


async def main() -> None:
    handle = build_live_provider()
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
            handle.provider,
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

        print_heading("Structured Extraction")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "valid": result.valid,
                "data": result.data,
                "repair_attempts": result.repair_attempts,
                "response_kind": result.response_kind,
                "validation_errors": result.validation_errors,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
