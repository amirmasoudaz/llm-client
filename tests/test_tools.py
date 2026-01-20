"""
Tests for the Tool system.
"""
import asyncio
import json
from typing import List, Optional
from unittest.mock import AsyncMock

import pytest

from llm_client.tools.base import Tool, ToolRegistry, ToolResult


class TestToolBasics:
    """Basic tool functionality tests."""
    
    def test_create_tool(self):
        """Test creating a tool manually."""
        async def handler(x: int) -> int:
            return x * 2
        
        tool = Tool(
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
        )
        
        assert tool.name == "double"
        assert tool.description == "Double a number"
    
    async def test_execute_tool(self):
        """Test executing a tool."""
        async def handler(message: str) -> str:
            return f"Received: {message}"
        
        tool = Tool(
            name="echo",
            description="Echo a message",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=handler,
        )
        
        result = await tool.execute(message="Hello")
        
        assert result.success
        assert "Received: Hello" in result.to_string()
    
    async def test_execute_json(self):
        """Test executing a tool with JSON arguments."""
        async def handler(x: int, y: int) -> int:
            return x + y
        
        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            handler=handler,
        )
        
        result = await tool.execute_json('{"x": 5, "y": 3}')
        
        assert result.success
        assert result.content == "8"
    
    async def test_execute_handles_error(self):
        """Test that tool execution errors are captured."""
        async def failing_handler():
            raise ValueError("Intentional failure")
        
        tool = Tool(
            name="fail",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            handler=failing_handler,
        )
        
        result = await tool.execute()
        
        assert not result.success
        assert "Intentional failure" in result.error
    
    def test_to_openai_format(self):
        """Test converting tool to OpenAI format."""
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            handler=AsyncMock(),
        )
        
        openai_format = tool.to_openai_format()
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "get_weather"
        assert openai_format["function"]["description"] == "Get weather for a city"


class TestToolDecorator:
    """Test the @tool decorator."""
    
    def test_tool_decorator_basic(self):
        """Test basic @tool decorator."""
        from llm_client import tool
        
        @tool
        async def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"
        
        assert greet.name == "greet"
        assert greet.description == "Greet someone by name."
        assert "name" in greet.parameters["properties"]
    
    def test_tool_decorator_with_defaults(self):
        """Test @tool decorator with default values."""
        from llm_client import tool
        
        @tool
        async def search(query: str, limit: int = 10) -> str:
            """Search for something."""
            return f"Found {limit} results for {query}"
        
        assert "query" in search.parameters["required"]
        # Limit should not be required since it has a default
        assert "limit" not in search.parameters.get("required", [])
    
    def test_sync_tool_decorator(self):
        """Test @sync_tool decorator for sync functions."""
        from llm_client import sync_tool
        
        @sync_tool
        def compute(x: int, y: int) -> int:
            """Compute x + y."""
            return x + y
        
        assert compute.name == "compute"
        assert "x" in compute.parameters["properties"]
        assert "y" in compute.parameters["properties"]
    
    async def test_sync_tool_executes_correctly(self):
        """Test sync tool executes in executor."""
        from llm_client import sync_tool
        import time
        
        @sync_tool
        def slow_sync(duration: float = 0.01) -> str:
            """Sleep then return."""
            time.sleep(duration)
            return "done"
        
        result = await slow_sync.execute(duration=0.01)
        
        assert result.success
        assert result.content == "done"


class TestToolRegistry:
    """Test the ToolRegistry class."""
    
    def test_create_registry(self):
        """Test creating an empty registry."""
        registry = ToolRegistry()
        
        assert len(registry) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        tool = Tool(
            name="test",
            description="Test",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        
        registry = ToolRegistry()
        registry.register(tool)
        
        assert len(registry) == 1
        assert "test" in registry
    
    def test_register_duplicate_fails(self):
        """Test that registering duplicate name raises error."""
        tool1 = Tool(
            name="dupe",
            description="First",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        tool2 = Tool(
            name="dupe",
            description="Second",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        
        registry = ToolRegistry()
        registry.register(tool1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        tool = Tool(
            name="removable",
            description="Can be removed",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        
        registry = ToolRegistry([tool])
        assert "removable" in registry
        
        removed = registry.unregister("removable")
        
        assert removed
        assert "removable" not in registry
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        tool = Tool(
            name="finder",
            description="Find me",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        
        registry = ToolRegistry([tool])
        found = registry.get("finder")
        
        assert found is tool
        assert registry.get("nonexistent") is None
    
    async def test_execute_by_name(self):
        """Test executing a tool by name."""
        async def handler(value: int) -> int:
            return value * 2
        
        tool = Tool(
            name="double",
            description="Double it",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            handler=handler,
        )
        
        registry = ToolRegistry([tool])
        result = await registry.execute("double", '{"value": 21}')
        
        assert result.success
        assert result.content == "42"
    
    async def test_execute_parallel(self):
        """Test parallel execution of multiple tools."""
        call_order = []
        
        async def tool_a():
            call_order.append("a_start")
            await asyncio.sleep(0.05)
            call_order.append("a_end")
            return "A"
        
        async def tool_b():
            call_order.append("b_start")
            await asyncio.sleep(0.05)
            call_order.append("b_end")
            return "B"
        
        registry = ToolRegistry([
            Tool(
                name="tool_a",
                description="A",
                parameters={"type": "object", "properties": {}},
                handler=tool_a,
            ),
            Tool(
                name="tool_b",
                description="B",
                parameters={"type": "object", "properties": {}},
                handler=tool_b,
            ),
        ])
        
        results = await registry.execute_parallel([
            {"name": "tool_a", "arguments": "{}"},
            {"name": "tool_b", "arguments": "{}"},
        ])
        
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Should be interleaved if parallel
        # Both should start before either ends
        assert call_order.index("b_start") < call_order.index("a_end")
    
    def test_to_openai_format(self):
        """Test converting all tools to OpenAI format."""
        registry = ToolRegistry([
            Tool(
                name="tool1",
                description="First tool",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock(),
            ),
            Tool(
                name="tool2",
                description="Second tool",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock(),
            ),
        ])
        
        openai_tools = registry.to_openai_format()
        
        assert len(openai_tools) == 2
        assert all(t["type"] == "function" for t in openai_tools)
    
    def test_tools_property(self):
        """Test tools property returns list."""
        tools = [
            Tool(
                name=f"tool{i}",
                description=f"Tool {i}",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock(),
            )
            for i in range(3)
        ]
        
        registry = ToolRegistry(tools)
        
        assert len(registry.tools) == 3
        assert registry.names == ["tool0", "tool1", "tool2"]


class TestToolResult:
    """Test the ToolResult class."""
    
    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult.success_result("done")
        
        assert result.success
        assert result.content == "done"
        assert result.error is None
    
    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error_result("Something went wrong")
        
        assert not result.success
        assert result.error == "Something went wrong"
    
    def test_to_string_success(self):
        """Test to_string for success."""
        result = ToolResult(content="The answer is 42", success=True)
        
        assert result.to_string() == "The answer is 42"
    
    def test_to_string_dict_content(self):
        """Test to_string for dict content."""
        result = ToolResult(content={"key": "value"}, success=True)
        
        stringified = result.to_string()
        assert "key" in stringified
        assert "value" in stringified
    
    def test_to_string_error(self):
        """Test to_string for error."""
        result = ToolResult(content=None, success=False, error="Failed")
        
        assert "Error: Failed" in result.to_string()
