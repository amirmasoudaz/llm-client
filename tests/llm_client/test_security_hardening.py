from __future__ import annotations

import pytest

from llm_client.observability import (
    LogFieldClass,
    ProviderPayloadCaptureMode,
    RedactionPolicy,
    ToolOutputPolicy,
    sanitize_log_data,
    sanitize_payload,
    sanitize_tool_output,
)
from llm_client.tools import ToolOutputPolicyMiddleware, ToolResult
from llm_client.tools.middleware import ToolExecutionContext
from tests.llm_client.test_cache_keys_and_diagnostics import _echo_tool, _search_tool


def test_redaction_policy_classifies_safe_sensitive_and_forbidden_fields() -> None:
    policy = RedactionPolicy()

    assert policy.classify_field("provider") is LogFieldClass.SAFE
    assert policy.classify_field("api_key") is LogFieldClass.SENSITIVE
    assert policy.classify_field("raw_response") is LogFieldClass.FORBIDDEN


def test_sanitize_log_data_omits_raw_provider_payloads_by_default() -> None:
    policy = RedactionPolicy()

    sanitized = sanitize_log_data(
        {
            "provider": "openai",
            "api_key": "sk-secret",
            "raw_response": {"id": "resp_1", "content": "sensitive body"},
        },
        policy=policy,
    )

    assert sanitized["provider"] == "openai"
    assert sanitized["api_key"] == "[REDACTED]"
    assert "raw_response" not in sanitized
    assert sanitized["_omitted_fields"] == ["raw_response"]


def test_provider_payload_capture_can_expose_metadata_only() -> None:
    policy = RedactionPolicy(provider_payload_capture=ProviderPayloadCaptureMode.METADATA_ONLY)

    sanitized = sanitize_log_data(
        {
            "raw_response": {"id": "resp_1", "model": "gpt-5", "content": "hidden", "status": 200},
        },
        policy=policy,
    )

    capture = sanitized["raw_response_capture"]
    assert capture["type"] == "dict"
    assert capture["id"] == "resp_1"
    assert capture["model"] == "gpt-5"
    assert capture["status"] == 200
    assert "content" not in capture


def test_sanitize_payload_omits_forbidden_nested_provider_payloads() -> None:
    policy = RedactionPolicy()

    sanitized = sanitize_payload(
        {
            "provider": "openai",
            "raw_response": {"id": "resp_1", "content": "hidden"},
        },
        policy=policy,
    )

    assert sanitized["provider"] == "openai"
    assert sanitized["raw_response"] == "<omitted>"


def test_tool_output_policy_redacts_and_truncates_output() -> None:
    sanitized = sanitize_tool_output(
        "email me at user@example.com with sk-secret-token " + ("x" * 120),
        policy=ToolOutputPolicy(max_chars=60),
    )

    assert "[REDACTED]" in sanitized
    assert "user@example.com" not in sanitized
    assert "sk-secret-token" not in sanitized
    assert sanitized.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_tool_output_policy_middleware_applies_hardening() -> None:
    middleware = ToolOutputPolicyMiddleware(policy=ToolOutputPolicy(max_chars=48))
    ctx = ToolExecutionContext(tool=_search_tool(), arguments={"query": "hello"})

    async def _next(_ctx: ToolExecutionContext) -> ToolResult:
        _ = _ctx
        return ToolResult.success_result("contact user@example.com with token=abc123 " + ("z" * 100))

    result = await middleware(ctx, _next)

    assert result.success is True
    assert "[REDACTED]" in (result.content or "")
    assert "user@example.com" not in (result.content or "")
    assert result.metadata["tool_output_policy_applied"] is True
