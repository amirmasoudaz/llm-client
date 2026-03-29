from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_client import Agent
from llm_client.agent import AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.providers.types import ToolCall
from llm_client.tools import Tool, ToolExecutionEngine, ToolExecutionMetadata, ToolRegistry, ToolResult


class _FakeProvider:
    model_name = "gpt-5-mini"
    model = SimpleNamespace(key="gpt-5-mini", count_tokens=lambda content: len(str(content or "").split()))


async def _echo_tool(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_tool_execution_engine_supports_all_execution_modes() -> None:
    registry = ToolRegistry(
        [
            Tool(name="one", description="one", parameters={"type": "object"}, handler=lambda: _echo_tool("one")),
            Tool(name="two", description="two", parameters={"type": "object"}, handler=lambda: _echo_tool("two")),
            Tool(name="three", description="three", parameters={"type": "object"}, handler=lambda: _echo_tool("three")),
        ]
    )
    engine = ToolExecutionEngine(registry)
    calls = [
        ToolCall(id="a", name="one", arguments="{}"),
        ToolCall(id="b", name="two", arguments="{}"),
        ToolCall(id="c", name="three", arguments="{}"),
    ]

    single = await engine.execute_calls(calls, mode=ToolExecutionMode.SINGLE)
    sequential = await engine.execute_calls(calls, mode=ToolExecutionMode.SEQUENTIAL)
    parallel = await engine.execute_calls(calls, mode=ToolExecutionMode.PARALLEL)
    planner = await engine.execute_calls(calls, mode=ToolExecutionMode.PLANNER)

    assert [result.status for result in single.results] == ["success", "skipped", "skipped"]
    assert [result.status for result in sequential.results] == ["success", "success", "success"]
    assert [result.status for result in parallel.results] == ["success", "success", "success"]
    assert [result.status for result in planner.results] == ["success", "skipped", "skipped"]
    assert planner.results[1].metadata["planner_managed"] is True


@pytest.mark.asyncio
async def test_tool_execution_engine_standardizes_result_envelopes_and_metadata() -> None:
    async def _partial() -> ToolResult:
        return ToolResult(content="trimmed", success=True, metadata={"partial": True})

    registry = ToolRegistry(
        [
            Tool(
                name="partial_tool",
                description="partial",
                parameters={"type": "object"},
                handler=_partial,
                execution=ToolExecutionMetadata(
                    timeout_seconds=12.0,
                    retry_attempts=2,
                    concurrency_limit=1,
                    safety_tags=("sensitive-read",),
                    trust_level="high",
                ),
            ),
        ]
    )
    engine = ToolExecutionEngine(registry)

    batch = await engine.execute_calls(
        [ToolCall(id="partial", name="partial_tool", arguments="{}")],
        mode=ToolExecutionMode.SEQUENTIAL,
    )
    result = batch.results[0]

    assert result.status == "partial"
    assert result.timeout_seconds == 12.0
    assert result.retry_attempts == 2
    assert result.concurrency_limit == 1
    assert result.safety_tags == ("sensitive-read",)
    assert result.trust_level == "high"
    assert result.to_tool_result().metadata["status"] == "partial"


@pytest.mark.asyncio
async def test_tool_execution_engine_retries_without_middleware() -> None:
    attempts = 0

    async def _flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return "ok"

    registry = ToolRegistry(
        [
            Tool(
                name="flaky",
                description="flaky",
                parameters={"type": "object"},
                handler=_flaky,
                execution=ToolExecutionMetadata(retry_attempts=1),
            ),
        ]
    )
    engine = ToolExecutionEngine(registry)

    batch = await engine.execute_calls(
        [ToolCall(id="retry", name="flaky", arguments="{}")],
        mode=ToolExecutionMode.SEQUENTIAL,
    )

    assert attempts == 2
    assert batch.results[0].status == "success"
    assert batch.results[0].attempts == 2


@pytest.mark.asyncio
async def test_agent_respects_explicit_single_tool_mode() -> None:
    registry = ToolRegistry(
        [
            Tool(name="one", description="one", parameters={"type": "object"}, handler=lambda: _echo_tool("one")),
            Tool(name="two", description="two", parameters={"type": "object"}, handler=lambda: _echo_tool("two")),
        ]
    )
    agent = Agent(
        provider=_FakeProvider(),
        tools=registry,
        definition=AgentDefinition(
            execution_policy=AgentExecutionPolicy(tool_execution_mode=ToolExecutionMode.SINGLE),
        ),
    )

    results = await agent._execute_tools(
        [
            ToolCall(id="a", name="one", arguments="{}"),
            ToolCall(id="b", name="two", arguments="{}"),
        ]
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].metadata["status"] == "skipped"
    assert "single_tool_mode" in (results[1].error or "")
