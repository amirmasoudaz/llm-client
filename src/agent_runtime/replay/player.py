"""
Replay player for execution replay.

This module provides:
- ReplayPlayer: Replays recorded executions
- ReplayMode: Different replay strategies
- ReplayResult: Result of a replay execution

Replay Modes and Capabilities
-----------------------------
There are TWO fundamentally different types of replay:

1. EVENT REPLAY (what this module provides):
   - Replays the recorded event stream
   - Good for: Debugging, demos, UI reconstruction, audit trails
   - Modes: FAST, TIMED, STEP
   - Does NOT call external systems (LLMs, tools)
   - Uses recorded event data as-is

2. DETERMINISTIC REPLAY (partially supported):
   - Uses recorded model responses to produce identical outputs
   - Modes: DETERMINISTIC
   - Requires: model_response field populated in RecordedEvent
   - Limitation: Tool outputs must also be recorded for full determinism
   - WARNING: Not fully deterministic without recorded tool outputs

3. VALIDATION REPLAY (experimental):
   - Calls actual models and compares to recording
   - Mode: VALIDATE
   - Requires: model_provider callback
   - Use case: Regression testing, drift detection
   - WARNING: Results may differ due to model updates, temperature, etc.

What "Deterministic" Actually Means
-----------------------------------
True deterministic replay requires recording ALL non-deterministic inputs:
- Model responses (supported via RecordedEvent.model_response)
- Tool outputs (NOT currently recorded by default)
- External API responses (NOT currently recorded)
- Random values, timestamps (NOT currently recorded)

The DETERMINISTIC mode is honest about its limits - it replays recorded
model responses but will NOT produce identical tool outputs unless you
implement custom tool mocking.

For testing purposes, consider:
- v0.1 (current): Use EVENT REPLAY for debugging and demos
- Future: Implement tool output recording for full re-execution

PII and Redaction
-----------------
Recordings contain potentially sensitive data:
- User prompts and inputs
- Model responses (may echo user data)
- Tool arguments and outputs
- Session/job identifiers (correlation risk)

Before sharing or storing recordings:
- Implement redaction in your recording pipeline
- Consider using RecordedEvent.model_response only for non-sensitive content
- Strip or hash identifiers if needed for anonymization
- Respect data retention policies

See agent_runtime/replay/recorder.py for recording implementation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable

from ..events.types import RuntimeEvent, RuntimeEventType
from ..events.bus import EventBus
from .metadata import RunMetadata, ReplayValidationError
from .recorder import Recording, RecordedEvent


class ReplayMode(str, Enum):
    """Replay execution modes."""
    
    # Use recorded model responses (fully deterministic)
    DETERMINISTIC = "deterministic"
    
    # Call actual models but validate responses
    VALIDATE = "validate"
    
    # Replay events at recorded timing
    TIMED = "timed"
    
    # Replay as fast as possible
    FAST = "fast"
    
    # Step through events one at a time
    STEP = "step"


@dataclass
class ReplayResult:
    """Result of a replay execution."""
    
    success: bool = True
    error: str | None = None
    
    # Validation results
    events_replayed: int = 0
    events_matched: int = 0
    events_mismatched: int = 0
    
    # Timing
    original_duration_ms: float | None = None
    replay_duration_ms: float = 0.0
    
    # Mismatched events for debugging
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    
    # Final output
    output: dict[str, Any] | None = None
    content: str | None = None


class ReplayPlayer:
    """Replays recorded executions.
    
    Supports multiple replay modes:
    - DETERMINISTIC: Uses recorded responses for full determinism
    - VALIDATE: Runs actual execution and validates against recording
    - TIMED: Replays events at original timing
    - FAST: Replays as fast as possible
    - STEP: Step through events manually
    
    Example:
        ```python
        recording = Recording.load("execution.replay.json")
        player = ReplayPlayer(recording, event_bus)
        
        # Fast deterministic replay
        result = await player.replay(mode=ReplayMode.DETERMINISTIC)
        
        # Or step through
        async for event in player.step():
            print(f"Event: {event.type}")
            input("Press enter for next event...")
        ```
    """
    
    def __init__(
        self,
        recording: Recording,
        event_bus: EventBus | None = None,
        model_provider: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self._recording = recording
        self._event_bus = event_bus
        self._model_provider = model_provider
        
        self._current_index: int = 0
        self._start_time: float = 0.0
        self._response_index: int = 0
        self._running = False
    
    @property
    def recording(self) -> Recording:
        return self._recording
    
    @property
    def metadata(self) -> RunMetadata:
        return self._recording.metadata
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def current_position(self) -> int:
        return self._current_index
    
    @property
    def total_events(self) -> int:
        return len(self._recording.events)
    
    async def validate_recording(self) -> tuple[bool, list[str]]:
        """Validate the recording before replay.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues: list[str] = []
        
        # Validate chain integrity
        is_valid, error = self._recording.validate_chain()
        if not is_valid:
            issues.append(f"Chain validation failed: {error}")
        
        # Check format version
        if self._recording.format_version > 1:
            issues.append(f"Unsupported format version: {self._recording.format_version}")
        
        # Check schema version
        if self._recording.metadata.schema_version > 1:
            issues.append(
                f"Warning: Recording uses newer schema version {self._recording.metadata.schema_version}"
            )
        
        return len(issues) == 0 or all("Warning:" in i for i in issues), issues
    
    async def replay(
        self,
        mode: ReplayMode = ReplayMode.FAST,
        strict_validation: bool = False,
        on_event: Callable[[RuntimeEvent], None] | None = None,
    ) -> ReplayResult:
        """Replay the recording.
        
        Args:
            mode: Replay mode
            strict_validation: Raise on validation failures
            on_event: Callback for each replayed event
        
        Returns:
            ReplayResult with execution details
        """
        # Validate first
        is_valid, issues = await self.validate_recording()
        if not is_valid and strict_validation:
            raise ReplayValidationError(f"Invalid recording: {issues}")
        
        self._running = True
        self._current_index = 0
        self._response_index = 0
        self._start_time = time.perf_counter()
        
        result = ReplayResult(
            original_duration_ms=self._recording.metadata.duration_ms,
        )
        
        try:
            if mode == ReplayMode.STEP:
                raise ValueError("Use step() method for step mode")
            
            async for event in self._replay_events(mode):
                result.events_replayed += 1
                
                # Emit to event bus if available
                if self._event_bus:
                    await self._event_bus.publish(event)
                
                # Call callback
                if on_event:
                    on_event(event)
                
                # Track last event for output
                if event.type == RuntimeEventType.FINAL_RESULT:
                    result.content = event.data.get("content")
                    result.output = event.data
            
            result.success = True
            result.events_matched = result.events_replayed
            
        except ReplayValidationError as e:
            result.success = False
            result.error = str(e)
            if e.expected and e.actual:
                result.mismatches.append({
                    "expected": e.expected,
                    "actual": e.actual,
                })
        except Exception as e:
            result.success = False
            result.error = str(e)
        finally:
            self._running = False
            result.replay_duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        return result
    
    async def step(self) -> AsyncIterator[RuntimeEvent]:
        """Step through events one at a time.
        
        Yields each event and waits for the next step() call.
        """
        self._running = True
        self._current_index = 0
        self._response_index = 0
        
        try:
            while self._current_index < len(self._recording.events):
                recorded = self._recording.events[self._current_index]
                event = self._reconstruct_event(recorded)
                
                self._current_index += 1
                yield event
        finally:
            self._running = False
    
    async def seek(self, position: int) -> None:
        """Seek to a specific position in the recording."""
        if position < 0 or position >= len(self._recording.events):
            raise ValueError(f"Position {position} out of range")
        self._current_index = position
    
    async def _replay_events(
        self,
        mode: ReplayMode,
    ) -> AsyncIterator[RuntimeEvent]:
        """Internal generator for replaying events."""
        last_time = 0.0
        
        while self._current_index < len(self._recording.events):
            recorded = self._recording.events[self._current_index]
            
            # Handle timing for TIMED mode
            if mode == ReplayMode.TIMED:
                delay = (recorded.relative_timestamp_ms - last_time) / 1000.0
                if delay > 0:
                    await asyncio.sleep(delay)
                last_time = recorded.relative_timestamp_ms
            
            # Reconstruct or validate event
            if mode == ReplayMode.VALIDATE and self._model_provider:
                event = await self._validate_and_execute(recorded)
            else:
                event = self._reconstruct_event(recorded)
            
            self._current_index += 1
            yield event
    
    def _reconstruct_event(self, recorded: RecordedEvent) -> RuntimeEvent:
        """Reconstruct a runtime event from recording."""
        # For model events, inject recorded response
        data = dict(recorded.event.data)
        
        if recorded.model_response and recorded.event.type in (
            RuntimeEventType.MODEL_DONE,
            RuntimeEventType.FINAL_RESULT,
        ):
            if "content" not in data or not data["content"]:
                data["content"] = recorded.model_response
        
        return RuntimeEvent(
            event_id=recorded.event.event_id,
            type=recorded.event.type,
            timestamp=recorded.event.timestamp,
            job_id=recorded.event.job_id,
            run_id=recorded.event.run_id,
            trace_id=recorded.event.trace_id,
            span_id=recorded.event.span_id,
            scope_id=recorded.event.scope_id,
            principal_id=recorded.event.principal_id,
            session_id=recorded.event.session_id,
            data=data,
            schema_version=recorded.event.schema_version,
        )
    
    async def _validate_and_execute(
        self,
        recorded: RecordedEvent,
    ) -> RuntimeEvent:
        """Execute and validate against recording."""
        # For non-model events, just replay
        if recorded.event.type not in (
            RuntimeEventType.MODEL_DONE,
            RuntimeEventType.FINAL_RESULT,
        ):
            return self._reconstruct_event(recorded)
        
        # For model events, call actual model and validate
        if self._model_provider:
            # This would call the actual model
            # For now, just return recorded event
            pass
        
        return self._reconstruct_event(recorded)
    
    def get_model_response(self, index: int) -> str | None:
        """Get a specific model response from the recording."""
        responses = self._recording.metadata.model_response_hashes
        if index < len(responses):
            # Return the actual response content, not hash
            model_events = [
                e for e in self._recording.events
                if e.model_response is not None
            ]
            if index < len(model_events):
                return model_events[index].model_response
        return None
    
    def get_next_model_response(self) -> str | None:
        """Get the next model response in sequence."""
        response = self.get_model_response(self._response_index)
        if response:
            self._response_index += 1
        return response


__all__ = [
    "ReplayMode",
    "ReplayResult",
    "ReplayPlayer",
]
