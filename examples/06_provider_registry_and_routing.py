from __future__ import annotations

import asyncio
import json
import os

from llm_client import (
    ExecutionEngine,
    Message,
    OpenAIProvider,
    AnthropicProvider,
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRegistry,
    RegistryRouter,
    RequestSpec,
    RetryConfig,
    load_env,
)

load_env()


def build_registry(
    primary_provider: OpenAIProvider,
    primary_model: str,
    secondary_provider: AnthropicProvider,
    secondary_model: str,
) -> ProviderRegistry:
    registry = ProviderRegistry()
    shared_capabilities = ProviderCapabilities(
        completions=True, streaming=True, embeddings=False, tool_calling=True
    )
    registry.register(
        ProviderDescriptor(
            name="primary_live",
            default_model=primary_model,
            priority=10,
            capabilities=shared_capabilities,
            metadata={"provider_family": "openai"},
            factory=lambda **_: primary_provider,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="secondary_live",
            default_model=secondary_model,
            priority=20,
            capabilities=shared_capabilities,
            metadata={"provider_family": "openai"},
            factory=lambda **_: secondary_provider,
        )
    )
    return registry


async def main() -> None:
    primary_model = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    secondary_model = os.getenv("LLM_CLIENT_EXAMPLE_SECONDARY_MODEL", "claude-3-5-haiku")
    primary_provider = OpenAIProvider(model=primary_model)
    secondary_provider = AnthropicProvider(model=secondary_model)
    try:
        registry = build_registry(primary_provider, primary_model, secondary_provider, secondary_model)
        router = RegistryRouter(registry=registry)
        engine = ExecutionEngine(router=router, retry=RetryConfig(attempts=1, backoff=0.0, max_backoff=0.0))
        result = await engine.complete(
            RequestSpec(
                provider="auto",
                model=primary_model,
                messages=[
                    Message.system("You are a helpful assistant that routes requests to the appropriate provider."),
                    Message.user(
                        "Route this request and answer in one sentence about registry-based routing."
                    ),
                ],
            )
        )

        print("\n=== Provider Registry ===\n")
        print(
            json.dumps(
                {
                    "providers": [descriptor.name for descriptor in registry.find_capable(completions=True)],
                    "primary": {"provider": "openai", "model": primary_model},
                    "secondary": {"provider": "anthropic", "model": secondary_model},
                },
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Routing + Live Completion ===\n")
        print(
            json.dumps(
                {
                    "status": result.status,
                    "model": result.model,
                    "content": result.content,
                },
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await primary_provider.close()
        if secondary_provider is not primary_provider:
            await secondary_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
