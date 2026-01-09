"""
Tool system for agent function calling.

This module provides:
- Tool dataclass for defining tools
- ToolRegistry for managing and executing tools
- ToolResult for standardized tool responses
"""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.
    
    Attributes:
        content: The tool's output (string or dict)
        success: Whether execution succeeded
        error: Error message if execution failed
        metadata: Additional data about execution
    """
    content: Union[str, Dict[str, Any], None] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert result to string for model consumption."""
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.content, dict):
            return json.dumps(self.content, indent=2)
        return str(self.content) if self.content is not None else ""
    
    @classmethod
    def success_result(cls, content: Union[str, Dict[str, Any]]) -> "ToolResult":
        """Create a successful result."""
        return cls(content=content, success=True)
    
    @classmethod
    def error_result(cls, error: str) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, error=error)


@dataclass
class Tool:
    """
    Definition of a callable tool for the agent.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description (shown to the model)
        parameters: JSON Schema defining the tool's parameters
        handler: Async function that executes the tool
        strict: Whether to enforce strict schema validation
    
    Example:
        ```python
        async def search_web(query: str, max_results: int = 5) -> str:
            # Implementation
            return f"Results for: {query}"
        
        search_tool = Tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            },
            handler=search_web
        )
        ```
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    strict: bool = False
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tools format."""
        tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
        if self.strict:
            tool_def["function"]["strict"] = True
        return tool_def
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given arguments.
        
        Returns:
            ToolResult with the execution outcome
        """
        try:
            result = await self.handler(**kwargs)
            
            # Normalize result to ToolResult
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult.success_result(result)
            else:
                return ToolResult.success_result(str(result))
                
        except Exception as e:
            return ToolResult.error_result(f"{type(e).__name__}: {str(e)}")
    
    async def execute_json(self, arguments_json: str) -> ToolResult:
        """
        Execute the tool with JSON-encoded arguments.
        
        Args:
            arguments_json: JSON string of arguments
            
        Returns:
            ToolResult with the execution outcome
        """
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            return await self.execute(**args)
        except json.JSONDecodeError as e:
            return ToolResult.error_result(f"Invalid JSON arguments: {e}")


class ToolRegistry:
    """
    Registry for managing and executing tools.
    
    Provides:
    - Tool registration and lookup
    - Conversion to API formats
    - Batch tool execution
    
    Example:
        ```python
        registry = ToolRegistry()
        registry.register(search_tool)
        registry.register(calculator_tool)
        
        # Execute a tool call from the model
        result = await registry.execute("search_web", '{"query": "python"}')
        
        # Get tools in OpenAI format
        tools = registry.to_openai_format()
        ```
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        """
        Initialize the registry.
        
        Args:
            tools: Optional list of tools to register initially
        """
        self._tools: Dict[str, Tool] = {}
        
        if tools:
            for tool in tools:
                self.register(tool)
    
    def register(self, tool: Tool) -> "ToolRegistry":
        """
        Register a tool.
        
        Args:
            tool: Tool to register
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return self
    
    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())
    
    @property
    def tools(self) -> List[Tool]:
        """Get list of all registered tools."""
        return list(self._tools.values())
    
    @property
    def names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    async def execute(
        self,
        name: str,
        arguments: Union[str, Dict[str, Any]],
    ) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Name of the tool to execute
            arguments: JSON string or dict of arguments
            
        Returns:
            ToolResult with execution outcome
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.error_result(f"Unknown tool: {name}")
        
        if isinstance(arguments, str):
            return await tool.execute_json(arguments)
        return await tool.execute(**arguments)
    
    async def execute_parallel(
        self,
        calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.
        
        Args:
            calls: List of dicts with 'name' and 'arguments' keys
            
        Returns:
            List of ToolResults in the same order
        """
        tasks = [
            self.execute(call["name"], call.get("arguments", {}))
            for call in calls
        ]
        return await asyncio.gather(*tasks)


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert Python type annotations to JSON Schema."""
    origin = getattr(py_type, "__origin__", None)
    
    # Handle Optional (Union with None)
    if origin is Union:
        args = py_type.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] -> X with nullable
            return _python_type_to_json_schema(non_none[0])
        # Union of multiple types
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}
    
    # Handle List/list
    if origin is list or (hasattr(py_type, "__origin__") and py_type.__origin__ is list):
        args = getattr(py_type, "__args__", (Any,))
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0]) if args else {}
        }
    
    # Handle Dict/dict
    if origin is dict or (hasattr(py_type, "__origin__") and py_type.__origin__ is dict):
        return {"type": "object"}
    
    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        type(None): {"type": "null"},
        Any: {},
    }
    
    return type_map.get(py_type, {"type": "string"})


def tool_from_function(
    func: Callable[..., Awaitable[Any]],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False,
) -> Tool:
    """
    Create a Tool from an async function.
    
    Uses function signature and docstring to generate tool definition.
    
    Args:
        func: Async function to convert
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        strict: Enable strict schema validation
        
    Returns:
        Tool instance
        
    Example:
        ```python
        async def get_weather(city: str, units: str = "celsius") -> str:
            '''Get current weather for a city.
            
            Args:
                city: Name of the city
                units: Temperature units (celsius or fahrenheit)
            '''
            return f"Weather in {city}: sunny"
        
        weather_tool = tool_from_function(get_weather)
        ```
    """
    if not asyncio.iscoroutinefunction(func):
        raise ValueError(f"Function {func.__name__} must be async")
    
    # Get function signature
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    
    # Build parameters schema
    properties: Dict[str, Any] = {}
    required: List[str] = []
    
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        
        param_type = hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(param_type)
        
        # Try to extract description from docstring
        if func.__doc__:
            # Simple extraction - look for "param_name:" in docstring
            for line in func.__doc__.split("\n"):
                line = line.strip()
                if line.startswith(f"{param_name}:"):
                    param_schema["description"] = line[len(param_name) + 1:].strip()
                    break
        
        properties[param_name] = param_schema
        
        # Required if no default
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    
    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    
    # Get description from docstring
    doc = description
    if not doc and func.__doc__:
        # Use first paragraph of docstring
        doc = func.__doc__.split("\n\n")[0].strip()
        # Remove Args: section if present
        if "\nArgs:" in doc:
            doc = doc.split("\nArgs:")[0].strip()
    
    return Tool(
        name=name or func.__name__,
        description=doc or f"Execute {func.__name__}",
        parameters=parameters,
        handler=func,
        strict=strict,
    )


__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool_from_function",
]

