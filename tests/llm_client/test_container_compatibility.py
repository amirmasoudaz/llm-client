from __future__ import annotations

import warnings

from llm_client.container import Container, create_agent, create_provider, get_container
from llm_client.engine import ExecutionEngine
from tests.llm_client.fakes import ScriptedProvider


def test_container_factory_helpers_warn_as_compatibility_surface(monkeypatch) -> None:
    import llm_client.providers.google as google_mod

    fake_provider = ScriptedProvider(model_name="gemini-2.0-flash")
    monkeypatch.setattr(google_mod, "GOOGLE_AVAILABLE", True)
    monkeypatch.setattr(google_mod, "GoogleProvider", lambda *args, **kwargs: fake_provider)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = create_provider("google", model="gemini-2.0-flash", api_key="test-key")

    assert provider is fake_provider
    assert any("compatibility/integration surface" in str(item.message) for item in caught)


def test_container_agent_factory_warns_as_compatibility_surface() -> None:
    engine = ExecutionEngine(provider=ScriptedProvider())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = create_agent(engine=engine, system_message="test")

    assert agent.engine is engine
    assert any("compatibility/integration surface" in str(item.message) for item in caught)


def test_global_container_helpers_warn_as_compatibility_surface() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        container = get_container()

    assert isinstance(container, Container)
    assert any("compatibility/integration surface" in str(item.message) for item in caught)
