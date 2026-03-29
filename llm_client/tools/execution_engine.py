"""
Dedicated tool execution engine with explicit execution modes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..agent.definition import ToolExecutionMode
from .base import ToolExecutionMetadata, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..providers.types import ToolCall
    from ..spec import RequestContext
    from .middleware import MiddlewareChain


ToolExecutionStatus = Literal["success", "partial", "error", "skipped"]


@dataclass
class ToolExecutionEnvelope:
    """Normalized record for one attempted tool execution."""

    tool_name: str
    tool_call_id: str
    arguments: str | dict[str, Any] | None = None
    result: ToolResult = field(default_factory=ToolResult)
    status: ToolExecutionStatus = "success"
    attempts: int = 1
    duration_ms: float | None = None
    timeout_seconds: float | None = None
    retry_attempts: int | None = None
    concurrency_limit: int | None = None
    safety_tags: tuple[str, ...] = ()
    trust_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"success", "partial"}

    def to_tool_result(self) -> ToolResult:
        merged_metadata = dict(self.result.metadata)
        merged_metadata.update(self.metadata)
        merged_metadata.setdefault("tool_name", self.tool_name)
        merged_metadata.setdefault("tool_call_id", self.tool_call_id)
        merged_metadata.setdefault("status", self.status)
        merged_metadata.setdefault("attempts", self.attempts)
        if self.duration_ms is not None:
            merged_metadata.setdefault("duration_ms", self.duration_ms)
        if self.timeout_seconds is not None:
            merged_metadata.setdefault("timeout_seconds", self.timeout_seconds)
        if self.retry_attempts is not None:
            merged_metadata.setdefault("retry_attempts", self.retry_attempts)
        if self.concurrency_limit is not None:
            merged_metadata.setdefault("concurrency_limit", self.concurrency_limit)
        if self.safety_tags:
            merged_metadata.setdefault("safety_tags", list(self.safety_tags))
        if self.trust_level is not None:
            merged_metadata.setdefault("trust_level", self.trust_level)
        return ToolResult(
            content=self.result.content,
            success=self.success,
            error=self.result.error,
            metadata=merged_metadata,
        )


@dataclass
class ToolExecutionBatch:
    """Execution batch result in input order."""

    mode: ToolExecutionMode
    results: list[ToolExecutionEnvelope] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(result.status == "error" for result in self.results)

    @property
    def has_partial(self) -> bool:
        return any(result.status == "partial" for result in self.results)


class ToolExecutionEngine:
    """Canonical tool execution engine for standalone llm_client usage."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        middleware_chain: MiddlewareChain | None = None,
        default_timeout: float = 30.0,
        default_retry_attempts: int = 0,
    ) -> None:
        self.registry = registry
        self.middleware_chain = middleware_chain
        self.default_timeout = default_timeout
        self.default_retry_attempts = default_retry_attempts

    async def execute_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        mode: ToolExecutionMode,
        request_context: RequestContext | None = None,
        max_tool_calls: int | None = None,
    ) -> ToolExecutionBatch:
        if not tool_calls:
            return ToolExecutionBatch(mode=mode, results=[])

        limited_calls = tool_calls[:max_tool_calls] if max_tool_calls is not None else list(tool_calls)

        if mode is ToolExecutionMode.SINGLE:
            results = [await self._execute_one(limited_calls[0], request_context=request_context)]
            for tool_call in limited_calls[1:]:
                results.append(self._skipped_result(tool_call, reason="single_tool_mode"))
            return ToolExecutionBatch(mode=mode, results=results)

        if mode is ToolExecutionMode.PLANNER:
            results = [await self._execute_one(limited_calls[0], request_context=request_context)]
            for tool_call in limited_calls[1:]:
                results.append(self._skipped_result(tool_call, reason="planner_managed"))
            return ToolExecutionBatch(mode=mode, results=results)

        if mode is ToolExecutionMode.SEQUENTIAL:
            results: list[ToolExecutionEnvelope] = []
            for tool_call in limited_calls:
                if request_context is not None:
                    request_context.cancellation_token.raise_if_cancelled()
                results.append(await self._execute_one(tool_call, request_context=request_context))
            return ToolExecutionBatch(mode=mode, results=results)

        semaphores: dict[str, asyncio.Semaphore] = {}

        async def _run_one(tool_call: ToolCall) -> ToolExecutionEnvelope:
            if request_context is not None:
                request_context.cancellation_token.raise_if_cancelled()
            tool = self.registry.get(str(getattr(tool_call, "name", "") or "").strip())
            limit = None if tool is None else self._resolve_metadata(tool.execution).concurrency_limit
            if not limit or limit < 1:
                return await self._execute_one(tool_call, request_context=request_context)
            semaphore = semaphores.setdefault(tool.name, asyncio.Semaphore(limit))
            async with semaphore:
                return await self._execute_one(tool_call, request_context=request_context)

        results = await asyncio.gather(*[_run_one(tool_call) for tool_call in limited_calls])
        return ToolExecutionBatch(mode=mode, results=list(results))

    @classmethod
    def from_agent_config(
        cls,
        registry: ToolRegistry,
        config: AgentConfig,
        *,
        middleware_chain: MiddlewareChain | None = None,
    ) -> ToolExecutionEngine:
        return cls(
            registry,
            middleware_chain=middleware_chain or config.get_middleware_chain(),
            default_timeout=config.tool_timeout,
            default_retry_attempts=config.tool_retry_attempts,
        )

    async def _execute_one(
        self,
        tool_call: ToolCall,
        *,
        request_context: RequestContext | None = None,
    ) -> ToolExecutionEnvelope:
        tool_name = str(getattr(tool_call, "name", "") or "").strip()
        tool_call_id = str(getattr(tool_call, "id", "") or "").strip() or str(uuid.uuid4())
        arguments = getattr(tool_call, "arguments", None)
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolExecutionEnvelope(
                tool_name=tool_name or "<unknown>",
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=ToolResult.error_result(f"Unknown tool: {tool_name or '<unknown>'}"),
                status="error",
            )

        execution_metadata = self._resolve_metadata(tool.execution)
        metadata = {
            "timeout": execution_metadata.timeout_seconds,
            "retry_attempts": execution_metadata.retry_attempts,
            "concurrency_limit": execution_metadata.concurrency_limit,
            "safety_tags": list(execution_metadata.safety_tags),
            "trust_level": execution_metadata.trust_level,
        }

        start = time.monotonic()
        if self.middleware_chain is None:
            result = await self._execute_without_middleware(
                tool_name,
                arguments or {},
                timeout_seconds=execution_metadata.timeout_seconds or self.default_timeout,
                retry_attempts=execution_metadata.retry_attempts or 0,
                request_context=request_context,
                metadata=metadata,
            )
        else:
            result = await self.registry.execute_with_middleware(
                tool_name,
                arguments or {},
                middleware_chain=self.middleware_chain,
                context=request_context,
                metadata=metadata,
            )
        duration_ms = (time.monotonic() - start) * 1000
        attempts = int(metadata.get("retry_attempts_used", 0)) + 1

        status: ToolExecutionStatus
        if not result.success:
            status = "error"
        elif result.metadata.get("partial") or result.metadata.get("truncated"):
            status = "partial"
        else:
            status = "success"

        return ToolExecutionEnvelope(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            result=result,
            status=status,
            attempts=attempts,
            duration_ms=duration_ms,
            timeout_seconds=execution_metadata.timeout_seconds,
            retry_attempts=execution_metadata.retry_attempts,
            concurrency_limit=execution_metadata.concurrency_limit,
            safety_tags=execution_metadata.safety_tags,
            trust_level=execution_metadata.trust_level,
            metadata={k: v for k, v in metadata.items() if k not in {"timeout"}},
        )

    def _resolve_metadata(self, metadata: ToolExecutionMetadata) -> ToolExecutionMetadata:
        return ToolExecutionMetadata(
            timeout_seconds=metadata.timeout_seconds if metadata.timeout_seconds is not None else self.default_timeout,
            retry_attempts=metadata.retry_attempts if metadata.retry_attempts is not None else self.default_retry_attempts,
            concurrency_limit=metadata.concurrency_limit,
            safety_tags=tuple(metadata.safety_tags),
            trust_level=metadata.trust_level,
        )

    async def _execute_without_middleware(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
        *,
        timeout_seconds: float,
        retry_attempts: int,
        request_context: RequestContext | None,
        metadata: dict[str, Any],
    ) -> ToolResult:
        last_error = None
        for attempt in range(retry_attempts + 1):
            if request_context is not None:
                request_context.cancellation_token.raise_if_cancelled()
            try:
                result = await asyncio.wait_for(
                    self.registry.execute(tool_name, arguments),
                    timeout=timeout_seconds,
                )
                metadata["retry_attempts_used"] = attempt
                if result.success:
                    return result
                last_error = result.error or "Tool execution failed"
            except asyncio.TimeoutError:
                last_error = f"Tool '{tool_name}' timed out after {timeout_seconds}s"
            except Exception as exc:
                last_error = f"Tool execution error: {exc}"

            if attempt < retry_attempts:
                await asyncio.sleep(1.0)

        metadata["retry_attempts_used"] = retry_attempts
        return ToolResult.error_result(last_error or "Unknown error")

    def _skipped_result(self, tool_call: ToolCall, *, reason: str) -> ToolExecutionEnvelope:
        return ToolExecutionEnvelope(
            tool_name=str(getattr(tool_call, "name", "") or "").strip() or "<unknown>",
            tool_call_id=str(getattr(tool_call, "id", "") or "").strip() or str(uuid.uuid4()),
            arguments=getattr(tool_call, "arguments", None),
            result=ToolResult(success=False, error=f"Tool skipped: {reason}"),
            status="skipped",
            metadata={"reason": reason, "planner_managed": reason == "planner_managed"},
        )


__all__ = [
    "ToolExecutionStatus",
    "ToolExecutionEnvelope",
    "ToolExecutionBatch",
    "ToolExecutionEngine",
]
