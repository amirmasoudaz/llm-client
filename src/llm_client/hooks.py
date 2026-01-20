"""
Lightweight hooks for observability and integration.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, Optional, Protocol


class Hook(Protocol):
    async def emit(self, event: str, payload: dict, context: Any) -> None:
        ...


class HookManager:
    def __init__(self, hooks: Optional[Iterable[Hook]] = None) -> None:
        self._hooks = list(hooks or [])

    def add(self, hook: Hook) -> None:
        self._hooks.append(hook)

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        for hook in self._hooks:
            result = hook.emit(event, payload, context)
            if asyncio.iscoroutine(result):
                await result


class InMemoryMetricsHook:
    """
    Simple metrics accumulator for tests and local inspection.
    """

    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.latencies_ms: list[int] = []

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        self.counters[event] = self.counters.get(event, 0) + 1
        if event == "request.end" and "latency_ms" in payload:
            self.latencies_ms.append(int(payload["latency_ms"]))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "latencies_ms": list(self.latencies_ms),
        }


class OpenTelemetryHook:
    """
    OpenTelemetry hook (optional dependency).
    """

    def __init__(self, tracer_name: str = "llm_client") -> None:
        try:
            from opentelemetry import trace
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("opentelemetry is required for OpenTelemetryHook") from exc

        self._trace = trace
        self._tracer = trace.get_tracer(tracer_name)
        self._spans: Dict[str, Any] = {}

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        request_id = getattr(context, "request_id", None)
        if event == "request.start" and request_id:
            span = self._tracer.start_span("llm.request")
            span.set_attribute("request.id", request_id)
            if "spec" in payload:
                span.set_attribute("llm.provider", payload["spec"].get("provider", ""))
                span.set_attribute("llm.model", payload["spec"].get("model", ""))
            self._spans[request_id] = span
        elif event == "request.end" and request_id:
            span = self._spans.pop(request_id, None)
            if span:
                span.set_attribute("llm.status", payload.get("status", 0))
                if "latency_ms" in payload:
                    span.set_attribute("llm.latency_ms", payload["latency_ms"])
                span.end()
        elif event == "provider.error" and request_id:
            span = self._spans.get(request_id)
            if span:
                span.add_event("provider.error", payload)


__all__ = ["Hook", "HookManager", "InMemoryMetricsHook", "OpenTelemetryHook"]
