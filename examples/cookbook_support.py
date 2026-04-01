from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client.config import load_env
from llm_client.providers import AnthropicProvider, GoogleProvider, OpenAIProvider


load_env()


CHAT_MODEL_DEFAULTS = {
    "openai": "gpt-5-nano",
    "anthropic": "claude-sonnet-4",
    "google": "gemini-2.0-flash",
}

SECONDARY_CHAT_MODEL_DEFAULTS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-opus-4",
    "google": "gemini-2.5-flash",
}

EMBEDDING_MODEL_DEFAULTS = {
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding-001",
}


@dataclass(frozen=True)
class LiveProviderHandle:
    name: str
    model: str
    provider: Any


def print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def require_optional_module(module_name: str, install_hint: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        print(f"Optional dependency '{module_name}' is not installed. {install_hint}")
        return False


def example_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def allow_skip() -> bool:
    return example_env("LLM_CLIENT_EXAMPLE_ALLOW_SKIP", "0") == "1"


def fail_or_skip(message: str) -> None:
    print(message)
    raise SystemExit(0 if allow_skip() else 1)


def require_database_dsn() -> str:
    dsn = example_env("LLM_CLIENT_EXAMPLE_PG_DSN")
    if not dsn:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PG_DSN to run this example against PostgreSQL.")
    return dsn


def _capability_provider_env(capability: str, *, secondary: bool) -> tuple[str, str]:
    if capability == "embeddings":
        return (
            "LLM_CLIENT_EXAMPLE_SECONDARY_EMBEDDINGS_PROVIDER" if secondary else "LLM_CLIENT_EXAMPLE_EMBEDDINGS_PROVIDER",
            "LLM_CLIENT_EXAMPLE_SECONDARY_EMBEDDINGS_MODEL" if secondary else "LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL",
        )
    return (
        "LLM_CLIENT_EXAMPLE_SECONDARY_PROVIDER" if secondary else "LLM_CLIENT_EXAMPLE_PROVIDER",
        "LLM_CLIENT_EXAMPLE_SECONDARY_MODEL" if secondary else "LLM_CLIENT_EXAMPLE_MODEL",
    )


def resolve_provider_name(*, capability: str = "chat", secondary: bool = False) -> str:
    provider_env, _ = _capability_provider_env(capability, secondary=secondary)
    default = "openai"
    provider_name = (example_env(provider_env, default) or default).strip().lower()
    if capability == "embeddings" and provider_name not in EMBEDDING_MODEL_DEFAULTS:
        fail_or_skip(
            "Embeddings examples require a provider with embedding support. "
            "Use openai or google via LLM_CLIENT_EXAMPLE_EMBEDDINGS_PROVIDER."
        )
    if capability != "embeddings" and provider_name not in CHAT_MODEL_DEFAULTS:
        fail_or_skip(
            "Unsupported example provider. Set LLM_CLIENT_EXAMPLE_PROVIDER to one of: "
            "openai, anthropic, google."
        )
    return provider_name


def resolve_model_name(
    provider_name: str,
    *,
    capability: str = "chat",
    secondary: bool = False,
) -> str:
    _, model_env = _capability_provider_env(capability, secondary=secondary)
    explicit = example_env(model_env)
    if explicit:
        return explicit
    if capability == "embeddings":
        return EMBEDDING_MODEL_DEFAULTS[provider_name]
    if secondary:
        return SECONDARY_CHAT_MODEL_DEFAULTS[provider_name]
    return CHAT_MODEL_DEFAULTS[provider_name]


def build_provider_handle(
    provider_name: str,
    model_name: str,
    *,
    use_responses_api: bool = False,
) -> LiveProviderHandle:
    provider_name = provider_name.strip().lower()
    if provider_name == "openai":
        if not example_env("OPENAI_API_KEY"):
            fail_or_skip("Set OPENAI_API_KEY to run the llm_client cookbook against OpenAI.")
        return LiveProviderHandle(
            name="openai",
            model=model_name,
            provider=OpenAIProvider(model=model_name, use_responses_api=use_responses_api),
        )
    if provider_name == "anthropic":
        if not example_env("ANTHROPIC_API_KEY"):
            fail_or_skip("Set ANTHROPIC_API_KEY to run the llm_client cookbook against Anthropic.")
        return LiveProviderHandle(
            name="anthropic",
            model=model_name,
            provider=AnthropicProvider(model=model_name),
        )
    if provider_name == "google":
        if not (example_env("GEMINI_API_KEY") or example_env("GOOGLE_API_KEY")):
            fail_or_skip("Set GEMINI_API_KEY or GOOGLE_API_KEY to run the llm_client cookbook against Google.")
        return LiveProviderHandle(
            name="google",
            model=model_name,
            provider=GoogleProvider(model=model_name),
        )
    fail_or_skip(f"Unsupported provider '{provider_name}'.")
    raise AssertionError("unreachable")


def build_live_provider(
    *,
    capability: str = "chat",
    secondary: bool = False,
    use_responses_api: bool = False,
) -> LiveProviderHandle:
    provider_name = resolve_provider_name(capability=capability, secondary=secondary)
    model_name = resolve_model_name(provider_name, capability=capability, secondary=secondary)
    if provider_name == "anthropic" and capability == "embeddings":
        fail_or_skip("Anthropic does not provide embeddings in llm_client. Use OpenAI or Google.")
    return build_provider_handle(provider_name, model_name, use_responses_api=use_responses_api)


async def close_provider(provider: Any) -> None:
    close_fn = getattr(provider, "close", None)
    if close_fn is None:
        return
    result = close_fn()
    if hasattr(result, "__await__"):
        await result


def summarize_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "total_cost": getattr(usage, "total_cost", None),
    }


__all__ = [
    "LiveProviderHandle",
    "allow_skip",
    "build_provider_handle",
    "build_live_provider",
    "close_provider",
    "fail_or_skip",
    "print_heading",
    "print_json",
    "require_database_dsn",
    "require_optional_module",
    "resolve_model_name",
    "resolve_provider_name",
    "summarize_usage",
]
