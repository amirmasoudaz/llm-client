"""
Tests for the Agent class with ReAct loop and tool execution.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.providers.types import (
    CompletionResult,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)


class TestAgentBasics:
    """Basic agent functionality tests."""
    
    async def test_agent_single_turn(self, mock_provider, mock_completion_result):
        """Test agent completing in a single turn."""
        from llm_client import Agent
        
        provider = mock_provider(responses=[
            mock_completion_result(content="Hello! How can I help?")
        ])
        
        agent = Agent(
            provider=provider,
            system_message="You are helpful.",
        )
        
        result = await agent.run("Hello!")
        
        assert result.status == "success"
        assert result.content == "Hello! How can I help?"
        assert result.num_turns == 1
        assert len(result.all_tool_calls) == 0
    
    async def test_agent_with_tool_call(
        self, 
        mock_provider, 
        mock_completion_result, 
        mock_tool_call,
        test_tool,
    ):
        """Test agent executing a tool and completing."""
        from llm_client import Agent
        
        tool_call = mock_tool_call(
            name="test_tool",
            arguments='{"message": "test"}'
        )
        
        provider = mock_provider(responses=[
            # First response: tool call
            mock_completion_result(
                content=None,
                tool_calls=[tool_call],
            ),
            # Second response: final answer
            mock_completion_result(content="Done! The tool returned: Echo: test"),
        ])
        
        agent = Agent(
            provider=provider,
            tools=[test_tool],
            system_message="You are helpful.",
        )
        
        result = await agent.run("Use the test tool")
        
        assert result.status == "success"
        assert result.num_turns == 2
        assert len(result.all_tool_calls) == 1
        assert result.all_tool_calls[0].name == "test_tool"
    
    async def test_agent_max_turns(self, mock_provider, mock_completion_result, mock_tool_call):
        """Test agent respecting max_turns limit."""
        from llm_client import Agent, Tool
        
        # Create a tool that the agent keeps calling
        infinite_tool = Tool(
            name="loop_tool",
            description="A tool",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(return_value="Keep going"),
        )
        
        tool_call = ToolCall(id="call_1", name="loop_tool", arguments="{}")
        
        # Agent always wants to call the tool
        provider = mock_provider(responses=[
            mock_completion_result(content=None, tool_calls=[tool_call])
            for _ in range(10)
        ])
        
        agent = Agent(
            provider=provider,
            tools=[infinite_tool],
            max_turns=3,
        )
        
        result = await agent.run("Loop forever")
        
        assert result.status == "max_turns"
        assert result.num_turns == 3
    
    async def test_agent_tool_error_handling(
        self,
        mock_provider,
        mock_completion_result,
        mock_tool_call,
        error_tool,
    ):
        """Test agent handling tool execution errors."""
        from llm_client import Agent
        
        tool_call = ToolCall(
            id="call_err", 
            name="error_tool", 
            arguments='{"should_fail": true}'
        )
        
        provider = mock_provider(responses=[
            mock_completion_result(content=None, tool_calls=[tool_call]),
            mock_completion_result(content="Tool failed, but I can continue."),
        ])
        
        agent = Agent(
            provider=provider,
            tools=[error_tool],
        )
        
        result = await agent.run("Use the error tool")
        
        # Agent should recover from tool error
        assert result.status == "success"
        assert result.num_turns == 2


class TestAgentConversation:
    """Test agent conversation management."""
    
    async def test_agent_continues_conversation(
        self,
        mock_provider,
        mock_completion_result,
    ):
        """Test agent maintaining conversation context."""
        from llm_client import Agent
        
        provider = mock_provider(responses=[
            mock_completion_result(content="My name is Assistant."),
            mock_completion_result(content="I said my name is Assistant."),
        ])
        
        agent = Agent(
            provider=provider,
            system_message="You are Assistant.",
        )
        
        # First turn
        result1 = await agent.run("What's your name?")
        assert "Assistant" in result1.content
        
        # Second turn - should remember context
        result2 = await agent.run("What did you just say?")
        assert "Assistant" in result2.content
        
        # Check conversation length
        assert len(agent.conversation) == 4  # 2 user + 2 assistant
    
    async def test_agent_reset(self, mock_provider, mock_completion_result):
        """Test agent conversation reset."""
        from llm_client import Agent
        
        provider = mock_provider(responses=[
            mock_completion_result(content="Response 1"),
            mock_completion_result(content="Response after reset"),
        ])
        
        agent = Agent(
            provider=provider,
            system_message="You are helpful.",
        )
        
        await agent.run("First message")
        assert len(agent.conversation) == 2
        
        agent.reset()
        assert len(agent.conversation) == 0
        
        await agent.run("Fresh start")
        assert len(agent.conversation) == 2
    
    async def test_agent_fork(self, mock_provider, mock_completion_result):
        """Test agent forking creates independent copy."""
        from llm_client import Agent
        
        provider = mock_provider(responses=[
            mock_completion_result(content="Original response"),
            mock_completion_result(content="Forked response"),
        ])
        
        original = Agent(
            provider=provider,
            system_message="You are helpful.",
        )
        
        await original.run("Setup context")
        
        forked = original.fork()
        
        # Forked agent should have same conversation history
        assert len(forked.conversation) == len(original.conversation)
        
        # But modifying one shouldn't affect the other
        original.reset()
        assert len(original.conversation) == 0
        assert len(forked.conversation) == 2


class TestAgentStreaming:
    """Test agent streaming functionality."""
    
    async def test_agent_stream_simple(self, mock_provider, mock_completion_result):
        """Test basic agent streaming."""
        from llm_client import Agent
        
        # Create provider with streaming support
        async def mock_stream(messages, **kwargs):
            yield StreamEvent(type=StreamEventType.TOKEN, data="Hello")
            yield StreamEvent(type=StreamEventType.TOKEN, data=" there!")
            yield StreamEvent(
                type=StreamEventType.DONE,
                data=mock_completion_result(content="Hello there!")
            )
        
        provider = mock_provider()
        provider.stream = mock_stream
        
        agent = Agent(provider=provider)
        
        tokens = []
        result = None
        
        async for event in agent.stream("Hi!"):
            if event.type == StreamEventType.TOKEN:
                tokens.append(event.data)
            elif event.type == StreamEventType.DONE:
                result = event.data
        
        assert tokens == ["Hello", " there!"]
        assert result is not None


class TestAgentParallelTools:
    """Test parallel tool execution."""
    
    async def test_parallel_tool_execution(
        self,
        mock_provider,
        mock_completion_result,
    ):
        """Test that multiple tools execute in parallel."""
        from llm_client import Agent, Tool
        
        call_times = []
        
        async def tracked_tool(name: str):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)
            return f"Result: {name}"
        
        tool = Tool(
            name="tracked_tool",
            description="A tracked tool",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=tracked_tool,
        )
        
        # Create two tool calls
        tool_calls = [
            ToolCall(id="call_1", name="tracked_tool", arguments='{"name": "A"}'),
            ToolCall(id="call_2", name="tracked_tool", arguments='{"name": "B"}'),
        ]
        
        provider = mock_provider(responses=[
            mock_completion_result(content=None, tool_calls=tool_calls),
            mock_completion_result(content="Done with both tools"),
        ])
        
        from llm_client.agent import AgentConfig
        
        agent = Agent(
            provider=provider,
            tools=[tool],
            config=AgentConfig(parallel_tool_execution=True),
        )
        
        await agent.run("Run both tools")
        
        # If parallel, they should start at nearly the same time
        if len(call_times) == 2:
            time_diff = abs(call_times[1] - call_times[0])
            # Allow some slack, but they should start within 50ms of each other
            assert time_diff < 0.05, f"Tools not parallel: {time_diff}s apart"
