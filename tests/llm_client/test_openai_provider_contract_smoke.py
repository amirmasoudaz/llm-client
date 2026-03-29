from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.openai import OpenAIProvider

from tests.llm_client.provider_contracts import (
    ProviderContractSpec,
    assert_provider_complete_contract,
    run_provider_contract_suite,
)


class _DummyLimiter:
    class _Ctx:
        def __init__(self) -> None:
            self.output_tokens = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def limit(self, *, tokens: int, requests: int):
        _ = (tokens, requests)
        return self._Ctx()


@pytest.mark.asyncio
async def test_openai_provider_complete_matches_shared_contract() -> None:
    provider = OpenAIProvider(model="gpt-5-mini", api_key="test-key")
    provider.limiter = _DummyLimiter()  # type: ignore[assignment]

    async def _fake_create(**kwargs):
        _ = kwargs
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(to_dict=lambda: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        return SimpleNamespace(choices=[choice], usage=usage)

    provider.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
    )

    result = await assert_provider_complete_contract(
        provider,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_contract_suite_can_run_against_fake_provider_fixture() -> None:
    from tests.llm_client.provider_contracts import ContractFakeProvider

    await run_provider_contract_suite(
        spec=ProviderContractSpec(name="fake-reuse", supports_embeddings=True),
        provider_factory=lambda: ContractFakeProvider(),
    )
