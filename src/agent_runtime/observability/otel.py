"""
OpenTelemetry integration for agent runtime.

Provides span hierarchy for agent executions:
- agent.job (parent): Full job execution
- agent.run (child): Single LLM interaction
- agent.tool (child): Tool execution
- agent.action (child): Human-in-the-loop action

Follows OpenTelemetry GenAI semantic conventions where applicable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode, Span
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    SpanKind = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    Span = Any  # type: ignore

from ..events.types import RuntimeEvent, RuntimeEventType
from ..events.bus import EventBus, EventSubscription


def _require_otel() -> None:
    """Raise ImportError if opentelemetry is not available."""
    if not OTEL_AVAILABLE:
        raise ImportError(
            "OpenTelemetry integration requires opentelemetry-api. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry adapter.
    
    Attributes:
        tracer_name: Name of the tracer
        service_name: Service name for spans
        capture_input: Whether to capture input content (may contain PII)
        capture_output: Whether to capture output content (may contain PII)
        capture_tool_args: Whether to capture tool arguments
        record_exceptions: Whether to record exceptions in spans
        sampling_rate: Sampling rate (1.0 = all traces)
    """
    tracer_name: str = "agent_runtime"
    service_name: str = "agent-runtime"
    capture_input: bool = False
    capture_output: bool = False
    capture_tool_args: bool = False
    record_exceptions: bool = True
    sampling_rate: float = 1.0


class OpenTelemetryAdapter:
    """OpenTelemetry adapter that subscribes to RuntimeEvents.
    
    Creates hierarchical spans for agent executions:
    - agent.job: Root span for the entire job
    - agent.run: Child span for LLM interactions
    - agent.tool: Child span for tool executions
    - agent.action: Child span for human-in-the-loop actions
    
    Example:
        ```python
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        
        # Set up OTel (application code)
        trace.set_tracer_provider(TracerProvider())
        
        # Create adapter
        adapter = OpenTelemetryAdapter(event_bus, config=OTelConfig())
        
        # Start listening to events
        await adapter.start()
        
        # ... run jobs ...
        
        # Stop when done
        await adapter.stop()
        ```
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: OTelConfig | None = None,
    ):
        _require_otel()
        
        self._config = config or OTelConfig()
        self._event_bus = event_bus
        self._tracer = trace.get_tracer(
            self._config.tracer_name,
            schema_url="https://opentelemetry.io/schemas/1.21.0",
        )
        
        # Track active spans by their correlation IDs
        self._job_spans: dict[str, Span] = {}
        self._run_spans: dict[str, Span] = {}
        self._tool_spans: dict[str, Span] = {}
        self._action_spans: dict[str, Span] = {}
        
        # Subscription handle
        self._subscription: EventSubscription | None = None
        self._running = False
    
    async def start(self) -> None:
        """Start listening to events."""
        if self._running:
            return
        
        self._subscription = await self._event_bus.subscribe(
            self._handle_event,
            event_types=None,  # Subscribe to all events
        )
        self._running = True
    
    async def stop(self) -> None:
        """Stop listening and end any open spans."""
        self._running = False
        
        if self._subscription:
            await self._subscription.unsubscribe()
            self._subscription = None
        
        # End any remaining spans
        for span in list(self._action_spans.values()):
            span.end()
        for span in list(self._tool_spans.values()):
            span.end()
        for span in list(self._run_spans.values()):
            span.end()
        for span in list(self._job_spans.values()):
            span.end()
        
        self._action_spans.clear()
        self._tool_spans.clear()
        self._run_spans.clear()
        self._job_spans.clear()
    
    async def _handle_event(self, event: RuntimeEvent) -> None:
        """Handle a runtime event by creating/updating spans."""
        try:
            handler = self._event_handlers.get(event.type)
            if handler:
                await handler(self, event)
        except Exception:
            # Don't let telemetry errors break the runtime
            pass
    
    async def _on_job_start(self, event: RuntimeEvent) -> None:
        """Handle job start event."""
        job_id = event.job_id
        if not job_id:
            return
        
        # Create root span for job
        span = self._tracer.start_span(
            "agent.job",
            kind=SpanKind.SERVER,
            attributes=self._job_attributes(event),
        )
        self._job_spans[job_id] = span
    
    async def _on_job_complete(self, event: RuntimeEvent) -> None:
        """Handle job complete event."""
        job_id = event.job_id
        if not job_id:
            return
        
        span = self._job_spans.pop(job_id, None)
        if span:
            # Add final attributes
            span.set_attribute("agent.job.status", event.data.get("status", "unknown"))
            if event.data.get("error"):
                span.set_status(Status(StatusCode.ERROR, event.data["error"]))
                span.set_attribute("agent.job.error", event.data["error"])
            else:
                span.set_status(Status(StatusCode.OK))
            
            # Add usage if available
            if "usage" in event.data:
                usage = event.data["usage"]
                span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
                span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
                span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))
            
            span.end()
    
    async def _on_job_failed(self, event: RuntimeEvent) -> None:
        """Handle job failure event."""
        job_id = event.job_id
        if not job_id:
            return
        
        span = self._job_spans.pop(job_id, None)
        if span:
            error = event.data.get("error", "Unknown error")
            span.set_status(Status(StatusCode.ERROR, error))
            span.set_attribute("agent.job.status", "failed")
            span.set_attribute("agent.job.error", error)
            
            if self._config.record_exceptions:
                span.record_exception(Exception(error))
            
            span.end()
    
    async def _on_progress(self, event: RuntimeEvent) -> None:
        """Handle progress event."""
        job_id = event.job_id
        if not job_id:
            return
        
        span = self._job_spans.get(job_id)
        if span:
            span.add_event(
                "progress",
                attributes={
                    "progress": event.data.get("progress", 0),
                    "turn": event.data.get("turn", 0),
                },
            )
    
    async def _on_model_token(self, event: RuntimeEvent) -> None:
        """Handle model token event - create/update run span."""
        run_id = event.run_id
        job_id = event.job_id
        if not run_id:
            return
        
        # Create run span if not exists (first token)
        if run_id not in self._run_spans:
            parent_span = self._job_spans.get(job_id) if job_id else None
            context = trace.set_span_in_context(parent_span) if parent_span else None
            
            span = self._tracer.start_span(
                "agent.run",
                kind=SpanKind.CLIENT,
                context=context,
                attributes={
                    "gen_ai.system": event.data.get("provider", ""),
                    "gen_ai.request.model": event.data.get("model", ""),
                    "agent.job.id": job_id or "",
                    "agent.run.id": run_id,
                },
            )
            self._run_spans[run_id] = span
    
    async def _on_model_done(self, event: RuntimeEvent) -> None:
        """Handle model done event - complete run span."""
        run_id = event.run_id
        if not run_id:
            return
        
        span = self._run_spans.pop(run_id, None)
        if span:
            # Add usage attributes
            if "usage" in event.data:
                usage = event.data["usage"]
                span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
                span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
                span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))
            
            if "latency_ms" in event.data:
                span.set_attribute("gen_ai.response.latency_ms", event.data["latency_ms"])
            
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    async def _on_tool_start(self, event: RuntimeEvent) -> None:
        """Handle tool execution start."""
        tool_call_id = event.data.get("tool_call_id", event.event_id)
        job_id = event.job_id
        run_id = event.run_id
        
        # Get parent span context (prefer run span, fallback to job span)
        parent_span = self._run_spans.get(run_id) if run_id else None
        if not parent_span:
            parent_span = self._job_spans.get(job_id) if job_id else None
        context = trace.set_span_in_context(parent_span) if parent_span else None
        
        attributes = {
            "agent.tool.name": event.data.get("tool_name", "unknown"),
            "agent.job.id": job_id or "",
            "agent.run.id": run_id or "",
        }
        
        if self._config.capture_tool_args and "arguments" in event.data:
            # Be careful with PII - only capture if explicitly enabled
            import json
            try:
                attributes["agent.tool.arguments"] = json.dumps(event.data["arguments"])
            except (TypeError, ValueError):
                pass
        
        span = self._tracer.start_span(
            "agent.tool",
            kind=SpanKind.CLIENT,
            context=context,
            attributes=attributes,
        )
        self._tool_spans[tool_call_id] = span
    
    async def _on_tool_end(self, event: RuntimeEvent) -> None:
        """Handle tool execution complete."""
        tool_call_id = event.data.get("tool_call_id", event.event_id)
        
        span = self._tool_spans.pop(tool_call_id, None)
        if span:
            if "duration_ms" in event.data:
                span.set_attribute("agent.tool.duration_ms", event.data["duration_ms"])
            if event.data.get("success", True):
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR))
            span.end()
    
    async def _on_tool_error(self, event: RuntimeEvent) -> None:
        """Handle tool execution error."""
        tool_call_id = event.data.get("tool_call_id", event.event_id)
        
        span = self._tool_spans.pop(tool_call_id, None)
        if span:
            error = event.data.get("error", "Tool error")
            span.set_status(Status(StatusCode.ERROR, error))
            span.set_attribute("agent.tool.error", error)
            
            if self._config.record_exceptions:
                span.record_exception(Exception(error))
            
            span.end()
    
    async def _on_action_required(self, event: RuntimeEvent) -> None:
        """Handle action required event."""
        action_id = event.data.get("action_id", event.event_id)
        job_id = event.job_id
        
        parent_span = self._job_spans.get(job_id) if job_id else None
        context = trace.set_span_in_context(parent_span) if parent_span else None
        
        span = self._tracer.start_span(
            "agent.action",
            kind=SpanKind.SERVER,
            context=context,
            attributes={
                "agent.action.id": action_id,
                "agent.action.type": event.data.get("type", "unknown"),
                "agent.job.id": job_id or "",
            },
        )
        self._action_spans[action_id] = span
    
    async def _on_action_resolved(self, event: RuntimeEvent) -> None:
        """Handle action resolved event."""
        action_id = event.data.get("action_id", event.event_id)
        
        span = self._action_spans.pop(action_id, None)
        if span:
            span.set_attribute("agent.action.status", "resolved")
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    async def _on_action_cancelled(self, event: RuntimeEvent) -> None:
        """Handle action cancelled event."""
        action_id = event.data.get("action_id", event.event_id)
        
        span = self._action_spans.pop(action_id, None)
        if span:
            span.set_attribute("agent.action.status", "cancelled")
            span.set_status(Status(StatusCode.ERROR, "Action cancelled"))
            span.end()
    
    def _job_attributes(self, event: RuntimeEvent) -> dict[str, Any]:
        """Build attributes for job span."""
        attrs: dict[str, Any] = {
            "agent.job.id": event.job_id or "",
            "agent.run.id": event.run_id or "",
            "service.name": self._config.service_name,
        }
        
        if event.scope_id:
            attrs["agent.scope.id"] = event.scope_id
        if event.principal_id:
            attrs["agent.principal.id"] = event.principal_id
        if event.session_id:
            attrs["agent.session.id"] = event.session_id
        
        # Add metadata from event data
        data = event.data
        if "model" in data:
            attrs["gen_ai.request.model"] = data["model"]
        if "provider" in data:
            attrs["gen_ai.system"] = data["provider"]
        
        return attrs
    
    # Event handler mapping
    _event_handlers = {
        RuntimeEventType.JOB_STARTED: _on_job_start,
        RuntimeEventType.JOB_COMPLETED: _on_job_complete,
        RuntimeEventType.JOB_FAILED: _on_job_failed,
        RuntimeEventType.PROGRESS: _on_progress,
        RuntimeEventType.MODEL_TOKEN: _on_model_token,
        RuntimeEventType.MODEL_DONE: _on_model_done,
        RuntimeEventType.TOOL_START: _on_tool_start,
        RuntimeEventType.TOOL_END: _on_tool_end,
        RuntimeEventType.TOOL_ERROR: _on_tool_error,
        RuntimeEventType.ACTION_REQUIRED: _on_action_required,
        RuntimeEventType.ACTION_RESOLVED: _on_action_resolved,
        RuntimeEventType.ACTION_CANCELLED: _on_action_cancelled,
    }


__all__ = [
    "OpenTelemetryAdapter",
    "OTelConfig",
]
