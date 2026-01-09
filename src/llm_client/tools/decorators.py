"""
Decorators for easy tool definition.

Provides the @tool decorator for converting functions to Tool instances.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable, Optional, TypeVar, overload

from .base import Tool, tool_from_function

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@overload
def tool(func: F) -> Tool: ...


@overload
def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False,
) -> Callable[[F], Tool]: ...


def tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False,
) -> Tool | Callable[[F], Tool]:
    """
    Decorator to convert an async function into a Tool.
    
    Can be used with or without arguments:
    
    ```python
    # Simple usage
    @tool
    async def search(query: str) -> str:
        '''Search for information.'''
        return f"Results for {query}"
    
    # With options
    @tool(name="web_search", description="Search the web")
    async def search(query: str, max_results: int = 5) -> str:
        return f"Results for {query}"
    ```
    
    The decorator extracts:
    - Parameter types from annotations
    - Parameter descriptions from docstring (Args: section)
    - Tool description from docstring first paragraph
    
    Args:
        func: The function to decorate
        name: Override tool name (default: function name)
        description: Override description (default: docstring)
        strict: Enable strict JSON schema validation
        
    Returns:
        Tool instance
    """
    def decorator(fn: F) -> Tool:
        return tool_from_function(
            fn,
            name=name,
            description=description,
            strict=strict,
        )
    
    if func is not None:
        return decorator(func)
    
    return decorator


def sync_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to convert a synchronous function into a Tool.
    
    The function will be wrapped to run in an executor.
    
    ```python
    @sync_tool
    def compute_hash(data: str) -> str:
        '''Compute SHA256 hash of data.'''
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    ```
    """
    import asyncio
    from functools import partial
    
    def decorator(fn: Callable[..., Any]) -> Tool:
        @wraps(fn)
        async def async_wrapper(**kwargs: Any) -> Any:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, partial(fn, **kwargs))
        
        # Copy metadata
        async_wrapper.__doc__ = fn.__doc__
        async_wrapper.__annotations__ = fn.__annotations__
        
        return tool_from_function(
            async_wrapper,
            name=name or fn.__name__,
            description=description,
            strict=strict,
        )
    
    if func is not None:
        return decorator(func)
    
    return decorator


__all__ = ["tool", "sync_tool"]

