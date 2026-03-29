from __future__ import annotations

import os

import pytest


def _skip_unless_enabled(flag: str, *env_vars: str) -> None:
    if os.getenv(flag) != "1":
        pytest.skip(f"{flag}=1 required for live provider integration test")
    missing = [name for name in env_vars if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing required env vars: {', '.join(missing)}")


@pytest.mark.asyncio
async def test_live_openai_complete_smoke() -> None:
    _skip_unless_enabled("LLM_CLIENT_LIVE_OPENAI", "OPENAI_API_KEY")

    from llm_client.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-5-mini")
    result = await provider.complete(
        "Reply with the word ok only.",
        max_tokens=64,
        reasoning_effort="minimal",
    )

    assert result.ok is True
    assert "ok" in str(result.content or "").strip().lower()


@pytest.mark.asyncio
async def test_live_anthropic_complete_smoke() -> None:
    _skip_unless_enabled("LLM_CLIENT_LIVE_ANTHROPIC", "ANTHROPIC_API_KEY")

    from llm_client.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-sonnet-4")
    result = await provider.complete("Reply with the word ok only.", max_tokens=16)

    assert result.ok is True
    assert result.content


@pytest.mark.asyncio
async def test_live_google_complete_smoke() -> None:
    _skip_unless_enabled("LLM_CLIENT_LIVE_GOOGLE", "GEMINI_API_KEY")

    from llm_client.providers.google import GoogleProvider

    provider = GoogleProvider(model="gemini-2.0-flash")
    result = await provider.complete("Reply with the word ok only.", max_tokens=16)

    assert result.ok is True
    assert result.content
