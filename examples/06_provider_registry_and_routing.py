from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, print_json

from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.providers.types import Message
from llm_client.routing import RegistryRouter
from llm_client.spec import RequestSpec


def build_registry(primary, secondary) -> ProviderRegistry:
    registry = ProviderRegistry()
    shared_capabilities = ProviderCapabilities(completions=True, streaming=True, embeddings=False, tool_calling=True)
    registry.register(
        ProviderDescriptor(
            name="primary_live",
            default_model=primary.model,
            priority=10,
            capabilities=shared_capabilities,
            metadata={"provider_family": primary.name},
            factory=lambda **_: primary.provider,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="secondary_live",
            default_model=secondary.model,
            priority=20,
            capabilities=shared_capabilities,
            metadata={"provider_family": secondary.name},
            factory=lambda **_: secondary.provider,
        )
    )
    return registry


async def main() -> None:
    primary = build_live_provider()
    secondary = build_live_provider(secondary=True)
    try:
        registry = build_registry(primary, secondary)
        router = RegistryRouter(registry=registry)
        engine = ExecutionEngine(router=router, retry=RetryConfig(attempts=1, backoff=0.0, max_backoff=0.0))
        result = await engine.complete(
            RequestSpec(
                provider="auto",
                model=primary.model,
                messages=[
                    Message.system("You are a helpful assistant that routes requests to the appropriate provider."),
                    Message.user("Route this request and answer in one sentence about registry-based routing.")
                ],
            )
        )

        print_heading("Provider Registry")
        print_json(
            {
                "providers": [descriptor.name for descriptor in registry.find_capable(completions=True)],
                "primary": {"provider": primary.name, "model": primary.model},
                "secondary": {"provider": secondary.name, "model": secondary.model},
            }
        )

        print_heading("Routing + Live Completion")
        print_json(
            {
                "status": result.status,
                "model": result.model,
                "content": result.content,
            }
        )
    finally:
        await close_provider(primary.provider)
        if secondary.provider is not primary.provider:
            await close_provider(secondary.provider)


if __name__ == "__main__":
    asyncio.run(main())
