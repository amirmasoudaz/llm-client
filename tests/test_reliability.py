from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.providers.base import BaseProvider
from llm_client.providers.types import (
    CompletionResult,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from llm_client.spec import RequestSpec


class MockProvider(BaseProvider):
    def __init__(self, model="mock", fail_count=0, fail_status=500):
        super().__init__(model)
        self.fail_count = fail_count
        self.current_fail = 0
        self.fail_status = fail_status
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.current_fail < self.fail_count:
            self.current_fail += 1
            if self.fail_status == -1:
                raise ValueError("Network error")
            return CompletionResult(status=self.fail_status, error="Simulated error")
        return CompletionResult(content="Success", status=200)

    async def stream(self, messages, **kwargs):
        self.calls += 1
        if self.current_fail < self.fail_count:
            self.current_fail += 1
            if self.fail_status == -1:
                raise ValueError("Network error")
            yield StreamEvent(type=StreamEventType.ERROR, data={"status": self.fail_status})
            return
        yield StreamEvent(type=StreamEventType.TOKEN, data="Success")
        yield StreamEvent(type=StreamEventType.DONE, data=CompletionResult(content="Success"))


@pytest.mark.asyncio
async def test_tool_retry_logic():
    # Setup
    mock_tool = AsyncMock()
    mock_tool.__name__ = "fail_once"
    mock_tool.return_value = "Result"

    # Fail first time, succeed second
    mock_tool.side_effect = [ValueError("Fail 1"), "Success"]

    # We need to register this tool with registry manually or use Agent
    # Testing execution.execute_single_tool directly might be easier
    from llm_client.agent.execution import execute_single_tool
    from llm_client.tools import Tool, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fail_once",
            description="Fail once",
            parameters={"type": "object", "properties": {}},
            handler=mock_tool,
        )
    )

    # Call with 0 retries -> should fail
    mock_tool.side_effect = [ValueError("Fail 1"), "Success"]
    result = await execute_single_tool(
        ToolCall(id="1", name="fail_once", arguments=""), registry, timeout=1.0, retries=0
    )
    assert not result.success
    assert "Fail 1" in result.error

    # Call with 1 retry -> should succeed
    mock_tool.reset_mock()
    mock_tool.side_effect = [ValueError("Fail 1"), "Success"]
    result = await execute_single_tool(
        ToolCall(id="1", name="fail_once", arguments=""), registry, timeout=1.0, retries=1
    )
    assert result.success
    assert result.content == "Success"


@pytest.mark.asyncio
async def test_engine_fallback_exception():
    # Test that engine falls back on exception
    p1 = MockProvider(model="gpt-5", fail_count=1, fail_status=-1)  # Raises ValueError
    p2 = MockProvider(model="gpt-5", fail_count=0)

    engine = ExecutionEngine(provider=p1, router=None)
    # Hack router to return p1 then p2
    engine.router = MagicMock()
    engine.router.select.return_value = [p1, p2]

    spec = RequestSpec(provider="mock", model="gpt-5", messages=[Message(role=Role.USER, content="hi")])

    result = await engine.complete(spec, retry=RetryConfig(attempts=1))
    assert result.ok
    assert result.content == "Success"
    assert p1.calls == 1
    assert p2.calls == 1


@pytest.mark.asyncio
async def test_engine_fallback_status():
    # Test that engine falls back on 500
    p1 = MockProvider(model="gpt-5", fail_count=1, fail_status=500)
    p2 = MockProvider(model="gpt-5", fail_count=0)

    engine = ExecutionEngine(provider=p1)
    engine.router = MagicMock()
    engine.router.select.return_value = [p1, p2]

    spec = RequestSpec(provider="mock", model="gpt-5", messages=[Message(role=Role.USER, content="hi")])

    result = await engine.complete(spec, retry=RetryConfig(attempts=1))
    assert result.ok
    assert p1.calls == 1
    assert p2.calls == 1
