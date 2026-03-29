from __future__ import annotations

import pytest

from llm_client.providers.types import StreamEventType

from tests.llm_client.provider_contracts import (
    ContractFakeProvider,
    ProviderContractSpec,
    run_provider_contract_suite,
)


@pytest.mark.asyncio
async def test_provider_contract_harness_supports_full_fake_provider() -> None:
    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="fake-full", supports_embeddings=True),
        provider_factory=lambda: ContractFakeProvider(supports_embeddings=True),
    )


@pytest.mark.asyncio
async def test_provider_contract_harness_supports_no_embedding_provider() -> None:
    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="fake-no-embed", supports_embeddings=False),
        provider_factory=lambda: ContractFakeProvider(supports_embeddings=False),
    )


@pytest.mark.asyncio
async def test_provider_contract_harness_enforces_terminal_stream_event() -> None:
    provider = ContractFakeProvider()

    events = [event async for event in provider.stream([{"role": "user", "content": "hello"}])]

    assert events[-1].type == StreamEventType.DONE
