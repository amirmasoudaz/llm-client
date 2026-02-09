# src/agents/orchestrator/tools.py
"""Tool registry and decorator for Dana orchestrator."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from functools import wraps

from pydantic import BaseModel, Field, create_model


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Complete tool definition."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    func: Optional[Callable] = None
    category: str = "general"
    requires_context: bool = True
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """
    Registry for all available tools in the Dana orchestrator.
    
    Tools can be registered via decorator or explicit registration.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
        category: str = "general",
        requires_context: bool = True,
    ) -> Callable:
        """Decorator to register a tool."""
        def decorator(func: Callable) -> Callable:
            # Extract parameters from function signature if not provided
            params = parameters or self._extract_parameters(func)
            
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=params,
                func=func,
                category=category,
                requires_context=requires_context,
            )
            
            self._tools[name] = tool_def
            
            # Track by category
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            
            wrapper._tool_definition = tool_def
            return wrapper
        
        return decorator
    
    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """Extract parameters from function signature."""
        params = []
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        
        # Skip certain parameter names
        skip_params = {'self', 'ctx', 'context', 'thread_id', 'student_id'}
        
        for name, param in sig.parameters.items():
            if name in skip_params:
                continue
            
            # Get type
            type_hint = hints.get(name, str)
            type_str = self._type_to_json_type(type_hint)
            
            # Get default
            has_default = param.default != inspect.Parameter.empty
            default = param.default if has_default else None
            
            # Get description from docstring (simplified)
            description = f"Parameter {name}"
            
            params.append(ToolParameter(
                name=name,
                type=type_str,
                description=description,
                required=not has_default,
                default=default,
            ))
        
        return params
    
    @staticmethod
    def _type_to_json_type(type_hint: Any) -> str:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            List: "array",
            Dict: "object",
        }
        
        # Handle Optional types
        if hasattr(type_hint, '__origin__'):
            origin = type_hint.__origin__
            if origin is Union:
                # Check for Optional (Union with None)
                args = type_hint.__args__
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return ToolRegistry._type_to_json_type(non_none[0])
            elif origin in type_map:
                return type_map[origin]
        
        return type_map.get(type_hint, "string")
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> Dict[str, ToolDefinition]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_by_category(self, category: str) -> List[ToolDefinition]:
        """Get tools by category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]
    
    def get_openai_tools(
        self,
        categories: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        tools = []
        exclude = exclude or []
        
        for name, tool in self._tools.items():
            if name in exclude:
                continue
            if categories and tool.category not in categories:
                continue
            tools.append(tool.to_openai_function())
        
        return tools
    
    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        if not tool.func:
            raise ValueError(f"Tool '{name}' has no implementation")
        
        # Add context if required
        if tool.requires_context and context is not None:
            arguments = {"ctx": context, **arguments}
        
        # Execute
        if asyncio.iscoroutinefunction(tool.func):
            return await tool.func(**arguments)
        return tool.func(**arguments)
    
    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())
    
    def describe(self) -> str:
        """Get a human-readable description of all tools."""
        lines = ["Available Tools:", ""]
        
        for category, names in self._categories.items():
            lines.append(f"## {category.title()}")
            for name in names:
                tool = self._tools.get(name)
                if tool:
                    lines.append(f"  - {name}: {tool.description}")
            lines.append("")
        
        return "\n".join(lines)


# Global tool registry instance
_global_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: Optional[List[ToolParameter]] = None,
    category: str = "general",
    requires_context: bool = True,
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Example:
        @tool(name="email_review", description="Review an outreach email")
        async def email_review(ctx, email_content: str) -> dict:
            ...
    """
    return _global_registry.register(
        name=name,
        description=description,
        parameters=parameters,
        category=category,
        requires_context=requires_context,
    )


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


# ============================================================================
# Tool Result Types
# ============================================================================

class ToolResult(BaseModel):
    """Base result for tool execution."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ReviewResult(ToolResult):
    """Result from a review tool."""
    score: Optional[float] = None
    dimensions: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    readiness_level: Optional[str] = None


class GenerateResult(ToolResult):
    """Result from a generation tool."""
    content: Optional[Dict[str, Any]] = None
    document_id: Optional[int] = None
    file_path: Optional[str] = None


class AlignmentResult(ToolResult):
    """Result from alignment evaluation."""
    score: float = 0.0
    label: str = "UNKNOWN"
    categories: Optional[List[Dict[str, Any]]] = None
    reasons: Optional[List[str]] = None


class ContextResult(ToolResult):
    """Result from context loading."""
    context_type: str = ""
    context: Optional[Dict[str, Any]] = None


class MemoryResult(ToolResult):
    """Result from memory operations."""
    memories: Optional[List[Dict[str, Any]]] = None
    memory_id: Optional[int] = None





