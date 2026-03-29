from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_client.agent.core import Agent
from llm_client.config.provider import AnthropicConfig, GoogleConfig, OpenAIConfig
from llm_client.container import Container, create_agent, create_provider
from llm_client.engine import ExecutionEngine


class _DummyProvider:
    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs


def test_provider_configs_use_supported_default_models() -> None:
    assert OpenAIConfig().default_model == "gpt-5"
    assert AnthropicConfig().default_model == "claude-sonnet-4"
    assert GoogleConfig().default_model == "gemini-2.0-flash"


def test_create_provider_supports_google(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)

    provider = create_provider("google", model="gemini-2.0-flash", api_key="test-key")

    assert isinstance(provider, _DummyProvider)
    assert provider.model == "gemini-2.0-flash"
    assert provider.api_key == "test-key"


def test_container_provider_supports_gemini_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)

    container = Container.default()
    provider = container.provider("gemini", model="gemini-2.0-flash", api_key="test-key")

    assert isinstance(provider, _DummyProvider)
    assert provider.model == "gemini-2.0-flash"


def test_create_provider_uses_default_registry_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)

    provider = create_provider("gemini", model="gemini-2.0-flash", api_key="test-key")

    assert isinstance(provider, _DummyProvider)
    assert provider.model == "gemini-2.0-flash"
    assert provider.api_key == "test-key"


@dataclass
class _EngineOnlyProvider:
    model: str = "gpt-5"


def test_create_agent_uses_supplied_engine_provider() -> None:
    engine_provider = _EngineOnlyProvider()
    engine = ExecutionEngine(provider=engine_provider)

    agent = create_agent(engine=engine, system_message="test")

    assert isinstance(agent, Agent)
    assert agent.engine is engine
    assert agent.provider is engine_provider


def test_agent_uses_engine_by_default() -> None:
    provider = _EngineOnlyProvider()

    agent = Agent(provider=provider)

    assert agent.provider is provider
    assert agent.engine is not None
    assert agent.engine.provider is provider


def test_agent_can_be_created_with_engine_only() -> None:
    engine_provider = _EngineOnlyProvider()
    engine = ExecutionEngine(provider=engine_provider)

    agent = Agent(engine=engine)

    assert agent.engine is engine
    assert agent.provider is engine_provider
