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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    TypeAlias,
    Union,
    cast,
    get_type_hints,
)

from ..concurrency import run_sync
from ..validation import validate_against_schema, validate_or_raise, validate_tool_definition


@dataclass(frozen=True)
class ToolExecutionMetadata:
    """Execution and safety hints attached to a tool definition."""

    timeout_seconds: float | None = None
    retry_attempts: int | None = None
    concurrency_limit: int | None = None
    safety_tags: tuple[str, ...] = ()
    trust_level: str | None = None


@dataclass(frozen=True)
class ResponsesGrammar:
    """Grammar descriptor for OpenAI Responses custom tools."""

    syntax: str
    definition: str
    type: str = "grammar"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "syntax": self.syntax,
            "definition": self.definition,
        }


@dataclass(frozen=True)
class ResponsesBuiltinTool:
    """Provider-native OpenAI Responses built-in tool descriptor."""

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            **dict(self.config),
        }

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def of(cls, tool_type: str, **config: Any) -> ResponsesBuiltinTool:
        return cls(type=tool_type, config=dict(config))

    @classmethod
    def web_search(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("web_search", **config)

    @classmethod
    def web_search_preview(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("web_search_preview", **config)

    @classmethod
    def file_search(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("file_search", **config)

    @classmethod
    def computer_use(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("computer_use", **config)

    @classmethod
    def code_interpreter(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("code_interpreter", **config)

    @classmethod
    def image_generation(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("image_generation", **config)

    @classmethod
    def mcp(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("mcp", **config)

    @classmethod
    def remote_mcp(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("mcp", **config)

    @classmethod
    def connector(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("mcp", **config)

    @classmethod
    def shell(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("shell", **config)

    @classmethod
    def apply_patch(cls, **config: Any) -> ResponsesBuiltinTool:
        return cls.of("apply_patch", **config)


@dataclass(frozen=True)
class ResponsesToolSearch:
    """Typed OpenAI Responses tool-search descriptor."""

    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_search",
            **dict(self.config),
        }

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def of(cls, **config: Any) -> ResponsesToolSearch:
        return cls(config=dict(config))

    @classmethod
    def hosted(cls, **config: Any) -> ResponsesToolSearch:
        return cls.of(**config)

    @classmethod
    def client(cls, **config: Any) -> ResponsesToolSearch:
        payload = dict(config)
        payload.setdefault("execution", "client")
        return cls.of(**payload)


class ResponsesConnectorId(str, Enum):
    """Documented OpenAI connector ids for MCP-backed connectors."""

    DROPBOX = "connector_dropbox"
    GMAIL = "connector_gmail"
    GOOGLE_CALENDAR = "connector_googlecalendar"
    GOOGLE_DRIVE = "connector_googledrive"
    MICROSOFT_TEAMS = "connector_microsoftteams"
    OUTLOOK_CALENDAR = "connector_outlookcalendar"
    OUTLOOK_EMAIL = "connector_outlookemail"
    SHAREPOINT = "connector_sharepoint"


@dataclass(frozen=True)
class ResponsesMCPToolFilter:
    """Allowed-tool filter for OpenAI MCP and connector tools."""

    tool_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_names": [name for name in self.tool_names if str(name).strip()],
        }


@dataclass(frozen=True)
class ResponsesMCPApprovalPolicy:
    """Approval policy object for OpenAI MCP and connector tools."""

    never: ResponsesMCPToolFilter | None = None
    always: ResponsesMCPToolFilter | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.never is not None:
            payload["never"] = self.never.to_dict()
        if self.always is not None:
            payload["always"] = self.always.to_dict()
        return payload


@dataclass(frozen=True)
class ResponsesMCPTool:
    """Typed OpenAI Responses MCP/connector descriptor."""

    server_label: str | None = None
    server_url: str | None = None
    connector_id: str | None = None
    server_description: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: tuple[str, ...] | None = None
    require_approval: str | ResponsesMCPApprovalPolicy | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "mcp"}
        if self.server_label:
            payload["server_label"] = self.server_label
        if self.server_url:
            payload["server_url"] = self.server_url
        if self.connector_id:
            payload["connector_id"] = self.connector_id
        if self.server_description:
            payload["server_description"] = self.server_description
        if self.authorization:
            payload["authorization"] = self.authorization
        if self.headers:
            payload["headers"] = dict(self.headers)
        if self.allowed_tools:
            payload["allowed_tools"] = [name for name in self.allowed_tools if str(name).strip()]
        if self.require_approval is not None:
            if isinstance(self.require_approval, ResponsesMCPApprovalPolicy):
                payload["require_approval"] = self.require_approval.to_dict()
            elif isinstance(self.require_approval, dict):
                payload["require_approval"] = dict(self.require_approval)
            else:
                payload["require_approval"] = self.require_approval
        payload.update(dict(self.metadata))
        return payload

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def remote_server(
        cls,
        server_url: str,
        *,
        server_label: str | None = None,
        server_description: str | None = None,
        authorization: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        require_approval: str | ResponsesMCPApprovalPolicy | dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ResponsesMCPTool:
        return cls(
            server_label=server_label,
            server_url=server_url,
            server_description=server_description,
            authorization=authorization,
            headers=dict(headers or {}) or None,
            allowed_tools=tuple(allowed_tools or ()) or None,
            require_approval=require_approval,
            metadata=dict(metadata),
        )

    @classmethod
    def connector(
        cls,
        connector_id: str | ResponsesConnectorId,
        *,
        server_label: str | None = None,
        authorization: str | None = None,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        require_approval: str | ResponsesMCPApprovalPolicy | dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ResponsesMCPTool:
        resolved_connector_id = (
            connector_id.value if isinstance(connector_id, ResponsesConnectorId) else str(connector_id)
        )
        return cls(
            connector_id=resolved_connector_id,
            server_label=server_label,
            authorization=authorization,
            allowed_tools=tuple(allowed_tools or ()) or None,
            require_approval=require_approval,
            metadata=dict(metadata),
        )

    @classmethod
    def deep_research_remote_server(
        cls,
        server_url: str,
        *,
        server_label: str | None = None,
        server_description: str | None = None,
        authorization: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        **metadata: Any,
    ) -> ResponsesMCPTool:
        return cls.remote_server(
            server_url,
            server_label=server_label,
            server_description=server_description,
            authorization=authorization,
            headers=headers,
            allowed_tools=allowed_tools,
            require_approval="never",
            **metadata,
        )

    @classmethod
    def deep_research_connector(
        cls,
        connector_id: str | ResponsesConnectorId,
        *,
        server_label: str | None = None,
        authorization: str | None = None,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        **metadata: Any,
    ) -> ResponsesMCPTool:
        return cls.connector(
            connector_id,
            server_label=server_label,
            authorization=authorization,
            allowed_tools=allowed_tools,
            require_approval="never",
            **metadata,
        )


@dataclass(frozen=True)
class ResponsesCustomTool:
    """Grammar-backed OpenAI Responses custom tool descriptor."""

    name: str
    description: str
    grammar: ResponsesGrammar
    strict: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "format": self.grammar.to_dict(),
        }
        if self.strict is not None:
            payload["strict"] = self.strict
        payload.update(dict(self.metadata))
        return payload

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class ResponsesFunctionTool:
    """Typed OpenAI Responses function descriptor with advanced provider metadata."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None
    defer_loading: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.strict is not None:
            payload["strict"] = self.strict
        if self.defer_loading is not None:
            payload["defer_loading"] = self.defer_loading
        payload.update(dict(self.metadata))
        return payload

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_tool(
        cls,
        tool: Tool,
        *,
        strict: bool | None = None,
        defer_loading: bool | None = None,
        **metadata: Any,
    ) -> ResponsesFunctionTool:
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=dict(tool.parameters),
            strict=(True if tool.strict else None) if strict is None else strict,
            defer_loading=defer_loading,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ResponsesToolNamespace:
    """Typed OpenAI Responses namespace descriptor for grouped function tools."""

    name: str
    description: str
    tools: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "namespace",
            "name": self.name,
            "description": self.description,
            "tools": [self._render_namespace_tool(tool) for tool in self.tools],
        }
        payload.update(dict(self.metadata))
        return payload

    def to_openai_format(self) -> dict[str, Any]:
        return self.to_dict()

    @staticmethod
    def _render_namespace_tool(tool: Any) -> dict[str, Any]:
        if isinstance(tool, dict):
            rendered = dict(tool)
        elif hasattr(tool, "to_openai_format"):
            rendered = tool.to_openai_format()
            if not isinstance(rendered, dict):
                raise ValueError("Namespace tools must render to dictionary definitions")
            rendered = dict(rendered)
        else:
            rendered = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            if getattr(tool, "strict", False):
                rendered["function"]["strict"] = True

        if str(rendered.get("type") or "") != "function":
            raise ValueError("OpenAI namespaces only support function tool definitions")

        function_payload = rendered.get("function")
        if isinstance(function_payload, dict):
            flattened = {
                key: function_payload[key]
                for key in ("name", "description", "parameters", "strict", "defer_loading")
                if key in function_payload
            }
            if flattened:
                rendered = {
                    "type": "function",
                    **flattened,
                }
        return rendered


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

    content: str | dict[str, Any] | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert result to string for model consumption."""
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.content, dict):
            return json.dumps(self.content, indent=2)
        return str(self.content) if self.content is not None else ""

    @classmethod
    def success_result(cls, content: str | dict[str, Any]) -> ToolResult:
        """Create a successful result."""
        return cls(content=content, success=True)

    @classmethod
    def error_result(cls, error: str) -> ToolResult:
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
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    strict: bool = False
    execution: ToolExecutionMetadata = field(default_factory=ToolExecutionMetadata)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tools format."""
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.strict:
            tool_def["function"]["strict"] = True
        return tool_def

    def _validate_arguments(self, args: dict[str, Any]) -> str | None:
        if not self.strict:
            return None

        result = validate_against_schema(args, self.parameters)
        if not result.valid:
            return "; ".join(result.errors)
        return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given arguments.

        Returns:
            ToolResult with the execution outcome
        """
        try:
            validation_error = self._validate_arguments(kwargs)
            if validation_error:
                return ToolResult.error_result(f"Tool validation failed: {validation_error}")

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


ToolDefinition: TypeAlias = (
    Tool
    | ResponsesBuiltinTool
    | ResponsesToolSearch
    | ResponsesFunctionTool
    | ResponsesToolNamespace
    | ResponsesMCPTool
    | ResponsesCustomTool
    | dict[str, Any]
)


def is_provider_native_tool(tool: Any) -> bool:
    """Return True for package-native provider tool descriptors."""
    return isinstance(
        tool,
        (
            ResponsesBuiltinTool,
            ResponsesToolSearch,
            ResponsesFunctionTool,
            ResponsesToolNamespace,
            ResponsesMCPTool,
            ResponsesCustomTool,
        ),
    )


def ensure_function_tools_only(
    tools: list[ToolDefinition] | None,
    *,
    provider: str,
) -> list[Any] | None:
    """Validate that a provider only receives executable function tools."""
    if not tools:
        return None
    invalid: list[str] = []
    for tool in tools:
        if isinstance(tool, Tool):
            continue
        if is_provider_native_tool(tool) or isinstance(tool, dict):
            invalid.append(type(tool).__name__)
            continue
        if all(hasattr(tool, attr) for attr in ("name", "description", "parameters")):
            continue
        invalid.append(type(tool).__name__)
    if invalid:
        unique = ", ".join(sorted(set(invalid)))
        raise ValueError(
            f"{provider} only supports executable Tool definitions; "
            f"provider-native or raw tool descriptors are not supported: {unique}"
        )
    return list(tools)


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

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """
        Initialize the registry.

        Args:
            tools: Optional list of tools to register initially
        """
        self._tools: dict[str, Tool] = {}

        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> ToolRegistry:
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

        # Validate tool definition structure
        res = validate_tool_definition(tool)
        validate_or_raise(res)

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

    def get(self, name: str) -> Tool | None:
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
    def tools(self) -> list[Tool]:
        """Get list of all registered tools."""
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: str | dict[str, Any],
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

    async def execute_with_middleware(
        self,
        name: str,
        arguments: str | dict[str, Any],
        *,
        middleware_chain: Any = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute a tool with middleware chain.

        Args:
            name: Name of the tool to execute
            arguments: JSON string or dict of arguments
            middleware_chain: Optional MiddlewareChain instance
            context: Optional RequestContext for correlation

        Returns:
            ToolResult with execution outcome
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.error_result(f"Unknown tool: {name}")

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult.error_result(f"Invalid JSON: {e}")
        else:
            args_dict = dict(arguments)

        # If no middleware, execute directly
        if middleware_chain is None:
            return await tool.execute(**args_dict)

        # Build execution context
        from .middleware import ToolExecutionContext

        exec_ctx = ToolExecutionContext(
            tool=tool,
            arguments=args_dict,
            request_context=context,
            metadata=dict(metadata or {}),
        )

        # Define the final handler (actual tool execution)
        async def final_handler(ctx: ToolExecutionContext) -> ToolResult:
            return await ctx.tool.execute(**ctx.arguments)

        # Build and execute the chain
        wrapped_handler = middleware_chain.build(final_handler)
        return await wrapped_handler(exec_ctx)

    async def execute_parallel(
        self,
        calls: list[dict[str, Any]],
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Args:
            calls: List of dicts with 'name' and 'arguments' keys

        Returns:
            List of ToolResults in the same order
        """
        tasks = [self.execute(call["name"], call.get("arguments", {})) for call in calls]
        return await asyncio.gather(*tasks)


def _python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
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
        return {"type": "array", "items": _python_type_to_json_schema(args[0]) if args else {}}

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
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> Tool:
    """
    Create a Tool from a function (sync or async).

    Uses function signature and docstring to generate tool definition.
    Synchronous functions are automatically wrapped to run in an executor.

    Args:
        func: Function to convert (sync or async)
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        strict: Enable strict schema validation

    Returns:
        Tool instance

    Example:
        ```python
        # Async function
        async def get_weather(city: str, units: str = "celsius") -> str:
            '''Get current weather for a city.

            Args:
                city: Name of the city
                units: Temperature units (celsius or fahrenheit)
            '''
            return f"Weather in {city}: sunny"

        # Sync function
        def calculate_hash(data: str) -> str:
            '''Calculate SHA256 hash.

            Args:
                data: Data to hash
            '''
            import hashlib
            return hashlib.sha256(data.encode()).hexdigest()

        weather_tool = tool_from_function(get_weather)
        hash_tool = tool_from_function(calculate_hash)
        ```
    """
    handler: Callable[..., Awaitable[Any]]
    func_for_schema: Callable[..., Any]

    # If sync function, wrap it to run in executor
    if not asyncio.iscoroutinefunction(func):
        from functools import wraps

        original_func = func

        @wraps(original_func)
        async def async_wrapper(**kwargs: Any) -> Any:
            return await run_sync(original_func, **kwargs)

        # Copy metadata for schema generation
        async_wrapper.__doc__ = original_func.__doc__
        async_wrapper.__annotations__ = original_func.__annotations__

        # Use the original function for signature inspection
        # but the wrapper as the handler
        func_for_schema = original_func
        handler = async_wrapper
    else:
        func_for_schema = func
        handler = cast(Callable[..., Awaitable[Any]], func)

    # Get function signature (use original for schema inspection)
    sig = inspect.signature(func_for_schema)
    hints = get_type_hints(func_for_schema) if hasattr(func_for_schema, "__annotations__") else {}

    # Build parameters schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(param_type)

        # Try to extract description from docstring
        if func_for_schema.__doc__:
            # Simple extraction - look for "param_name:" in docstring
            for line in func_for_schema.__doc__.split("\n"):
                line = line.strip()
                if line.startswith(f"{param_name}:"):
                    param_schema["description"] = line[len(param_name) + 1 :].strip()
                    break

        properties[param_name] = param_schema

        # Required if no default
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required

    # Get description from docstring
    doc = description
    if not doc and func_for_schema.__doc__:
        # Use first paragraph of docstring
        doc = func_for_schema.__doc__.split("\n\n")[0].strip()
        # Remove Args: section if present
        if "\nArgs:" in doc:
            doc = doc.split("\nArgs:")[0].strip()

    return Tool(
        name=name or func_for_schema.__name__,
        description=doc or f"Execute {func_for_schema.__name__}",
        parameters=parameters,
        handler=handler,
        strict=strict,
    )


__all__ = [
    "Tool",
    "ToolExecutionMetadata",
    "ToolResult",
    "ToolRegistry",
    "ResponsesBuiltinTool",
    "ResponsesToolSearch",
    "ResponsesConnectorId",
    "ResponsesFunctionTool",
    "ResponsesToolNamespace",
    "ResponsesMCPToolFilter",
    "ResponsesMCPApprovalPolicy",
    "ResponsesMCPTool",
    "ResponsesCustomTool",
    "ResponsesGrammar",
    "ToolDefinition",
    "is_provider_native_tool",
    "ensure_function_tools_only",
    "tool_from_function",
]
