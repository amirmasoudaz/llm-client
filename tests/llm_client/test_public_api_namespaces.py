import llm_client
import warnings

from llm_client import adapters, budgets, compat, config, content, engine, models, observability, providers, tools, types, validation
from llm_client.compat import OpenAIClient, ResponseTimeoutError
from llm_client.adapters import PostgresSQLAdaptor, SQLQueryRequest
from llm_client.content import Message, Role, ToolCall, normalize_messages
from llm_client.budgets import Budget, Ledger
from llm_client.observability import (
    BenchmarkRecorder,
    EngineDiagnosticsRecorder,
    EngineDiagnosticsSnapshot,
    InMemoryEventBus,
    LatencyRecorder,
    LifecycleLoggingHook,
    ReplayRecorder,
    RedactionPolicy,
    RuntimeEvent,
    RuntimeEventType,
)
from llm_client.types import (
    CancellationToken,
    CompletionResult,
    RequestContext,
    RequestSpec,
    StreamEvent,
    StreamEventType,
    Usage,
)


def test_content_namespace_exports_core_message_types() -> None:
    tool_call = ToolCall(id="call_1", name="lookup", arguments='{"q":"hello"}')
    message = Message.assistant(tool_calls=[tool_call])

    assert message.role is Role.ASSISTANT
    assert normalize_messages([message])[0].tool_calls[0].name == "lookup"


def test_types_namespace_exports_shared_request_and_result_types() -> None:
    spec = RequestSpec(
        provider="openai",
        model="gpt-5",
        messages=[Message.user("hi")],
    )
    result = CompletionResult(content="hello", usage=Usage(total_tokens=3))
    event = StreamEvent(type=StreamEventType.DONE, data=result)

    assert spec.provider == "openai"
    assert RequestContext(session_id="session-1").session_id == "session-1"
    assert result.usage.total_tokens == 3
    assert event.type.value == "done"
    assert isinstance(CancellationToken.none(), CancellationToken)


def test_observability_namespace_exports_runtime_and_replay_primitives() -> None:
    bus = InMemoryEventBus()
    recorder = ReplayRecorder(event_bus=bus)
    runtime_event = RuntimeEvent(type=RuntimeEventType.PROGRESS, data={"step": "start"})
    diagnostics = EngineDiagnosticsSnapshot(kind="request", request_id="req-1", payload={"status": 200})

    assert recorder.is_recording is False
    assert runtime_event.type is RuntimeEventType.PROGRESS
    assert isinstance(LatencyRecorder(), LatencyRecorder)
    assert isinstance(EngineDiagnosticsRecorder(), EngineDiagnosticsRecorder)
    assert isinstance(BenchmarkRecorder(), BenchmarkRecorder)
    assert isinstance(LifecycleLoggingHook(), LifecycleLoggingHook)
    assert isinstance(RedactionPolicy(), RedactionPolicy)
    assert diagnostics.request_id == "req-1"


def test_compat_namespace_exposes_legacy_client_api() -> None:
    assert OpenAIClient is not None
    assert issubclass(ResponseTimeoutError, TimeoutError)


def test_top_level_compat_aliases_warn_and_resolve() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_client = llm_client.OpenAIClient

    assert legacy_client is OpenAIClient
    assert any("compatibility-only" in str(item.message) for item in caught)


def test_top_level_all_is_now_a_smaller_stable_contract() -> None:
    assert "Agent" in llm_client.__all__
    assert "OpenAIProvider" in llm_client.__all__
    assert "OpenAIClient" in llm_client.__all__
    assert "TelemetryCounter" not in llm_client.__all__
    assert "fingerprint" not in llm_client.__all__
    assert not hasattr(llm_client, "TelemetryCounter")
    assert not hasattr(llm_client, "fingerprint")
    assert not hasattr(llm_client, "BatchManager")


def test_public_modules_define_explicit_all_contracts() -> None:
    assert "SQLQueryRequest" in adapters.__all__
    assert "ExecutionEngine" in engine.__all__
    assert "Ledger" in budgets.__all__
    assert "Message" in content.__all__
    assert "OpenAIProvider" in providers.__all__
    assert "ToolRegistry" in tools.__all__
    assert "RequestContext" in types.__all__
    assert "RuntimeEvent" in observability.__all__
    assert "ValidationError" in validation.__all__
    assert "load_env" in config.__all__
    assert "GPT5" in models.__all__
    assert "OpenAIClient" in compat.__all__


def test_budget_namespace_exports_generic_usage_and_budget_primitives() -> None:
    assert Ledger is not None
    assert Budget is not None


def test_adapter_namespace_exports_generic_service_contracts() -> None:
    request = SQLQueryRequest(statement="select 1")

    assert request.statement == "select 1"
    assert PostgresSQLAdaptor.backend_name == "postgres"


def test_public_module_all_contracts_avoid_internal_helpers() -> None:
    assert "_TOOL_SCHEMA_STRICT" not in validation.__all__
    assert "_default_agent_config" not in config.__all__
    assert "_DEFAULT_MAX_TOOL_OUTPUT_CHARS" not in tools.__all__
