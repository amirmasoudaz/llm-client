"""
Replay recorder for capturing execution events.

This module provides:
- RecordedEvent: A single recorded event
- Recording: A complete recording of an execution
- ReplayRecorder: Records events from the event bus
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from ..events.types import RuntimeEvent, RuntimeEventType
from ..events.bus import EventBus, EventSubscription
from .metadata import RunMetadata, EventFingerprint


@dataclass
class RecordedEvent:
    """A single recorded event with fingerprint.
    
    Contains:
    - The original runtime event
    - A fingerprint for validation
    - Recording-specific metadata (relative timestamp)
    """
    event: RuntimeEvent
    fingerprint: EventFingerprint
    relative_timestamp_ms: float
    
    # For model events, capture the full response
    model_response: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "fingerprint": self.fingerprint.to_dict(),
            "relative_timestamp_ms": self.relative_timestamp_ms,
            "model_response": self.model_response,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordedEvent:
        return cls(
            event=RuntimeEvent.from_dict(data["event"]),
            fingerprint=EventFingerprint.from_dict(data["fingerprint"]),
            relative_timestamp_ms=data["relative_timestamp_ms"],
            model_response=data.get("model_response"),
        )


@dataclass
class Recording:
    """A complete recording of an execution.
    
    Contains:
    - Metadata about the run
    - All recorded events in order
    - Initial input that started the execution
    """
    metadata: RunMetadata
    events: list[RecordedEvent] = field(default_factory=list)
    initial_input: dict[str, Any] = field(default_factory=dict)
    
    # File format version
    format_version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "metadata": self.metadata.to_dict(),
            "initial_input": self.initial_input,
            "events": [e.to_dict() for e in self.events],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recording:
        return cls(
            format_version=data.get("format_version", 1),
            metadata=RunMetadata.from_dict(data["metadata"]),
            initial_input=data.get("initial_input", {}),
            events=[RecordedEvent.from_dict(e) for e in data.get("events", [])],
        )
    
    def save(self, path: str | Path) -> None:
        """Save recording to a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Load recording from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_binary(self, f: BinaryIO) -> None:
        """Save recording in compact binary format."""
        import gzip
        data = json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")
        f.write(gzip.compress(data))
    
    @classmethod
    def load_binary(cls, f: BinaryIO) -> Recording:
        """Load recording from binary format."""
        import gzip
        data = json.loads(gzip.decompress(f.read()).decode("utf-8"))
        return cls.from_dict(data)
    
    def validate_chain(self) -> tuple[bool, str | None]:
        """Validate the event chain integrity.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.events:
            return True, None
        
        # Check first event hash matches metadata
        if (
            self.metadata.first_event_hash
            and self.events[0].fingerprint.hash != self.metadata.first_event_hash
        ):
            return False, "First event hash mismatch"
        
        # Check chain continuity
        prev_hash = None
        for i, recorded in enumerate(self.events):
            if recorded.fingerprint.parent_hash != prev_hash:
                return False, f"Chain broken at event {i}"
            prev_hash = recorded.fingerprint.hash
        
        # Check last event hash
        if (
            self.metadata.last_event_hash
            and prev_hash != self.metadata.last_event_hash
        ):
            return False, "Last event hash mismatch"
        
        return True, None
    
    def get_events_by_type(
        self,
        event_type: RuntimeEventType,
    ) -> list[RecordedEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event.type == event_type]
    
    def get_model_responses(self) -> list[str]:
        """Get all recorded model responses."""
        return [
            e.model_response
            for e in self.events
            if e.model_response is not None
        ]


class ReplayRecorder:
    """Records execution events for later replay.
    
    Subscribes to the event bus and captures all events along with
    fingerprints for validation.
    
    Example:
        ```python
        recorder = ReplayRecorder(event_bus, metadata)
        
        # Start recording
        await recorder.start(initial_input={"prompt": "Hello"})
        
        # ... execution happens ...
        
        # Stop and get recording
        recording = await recorder.stop()
        recording.save("execution.replay.json")
        ```
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        metadata: RunMetadata | None = None,
        capture_model_responses: bool = True,
    ):
        self._event_bus = event_bus
        self._metadata = metadata or RunMetadata()
        self._capture_responses = capture_model_responses
        
        self._recording: Recording | None = None
        self._subscription: EventSubscription | None = None
        self._start_time: float = 0.0
        self._sequence: int = 0
        self._last_fingerprint: EventFingerprint | None = None
        self._lock = asyncio.Lock()
        self._running = False
    
    @property
    def is_recording(self) -> bool:
        return self._running
    
    async def start(
        self,
        initial_input: dict[str, Any] | None = None,
    ) -> None:
        """Start recording events."""
        if self._running:
            return
        
        async with self._lock:
            self._recording = Recording(
                metadata=self._metadata,
                initial_input=initial_input or {},
            )
            self._start_time = time.perf_counter()
            self._sequence = 0
            self._last_fingerprint = None
            
            # Subscribe to all events
            self._subscription = await self._event_bus.subscribe(
                self._record_event,
                event_types=None,
            )
            self._running = True
    
    async def stop(self) -> Recording:
        """Stop recording and return the recording."""
        if not self._running:
            raise RuntimeError("Recorder is not running")
        
        async with self._lock:
            self._running = False
            
            if self._subscription:
                await self._subscription.unsubscribe()
                self._subscription = None
            
            # Finalize metadata
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._metadata.duration_ms = duration_ms
            self._metadata.event_count = self._sequence
            
            recording = self._recording
            self._recording = None
            
            return recording
    
    async def _record_event(self, event: RuntimeEvent) -> None:
        """Record a single event."""
        if not self._running or not self._recording:
            return
        
        async with self._lock:
            # Calculate relative timestamp
            relative_ts = (time.perf_counter() - self._start_time) * 1000
            
            # Compute fingerprint
            parent_hash = self._last_fingerprint.hash if self._last_fingerprint else None
            fingerprint = EventFingerprint.compute(
                event_type=event.type.value,
                event_data=event.data,
                sequence=self._sequence,
                parent_hash=parent_hash,
            )
            
            # Extract model response if applicable
            model_response = None
            if self._capture_responses and event.type in (
                RuntimeEventType.MODEL_DONE,
                RuntimeEventType.FINAL_RESULT,
            ):
                model_response = event.data.get("content") or event.data.get("response")
                if model_response:
                    self._metadata.add_model_response(model_response)
            
            # Create recorded event
            recorded = RecordedEvent(
                event=event,
                fingerprint=fingerprint,
                relative_timestamp_ms=relative_ts,
                model_response=model_response,
            )
            
            self._recording.events.append(recorded)
            self._metadata.update_event_chain(fingerprint)
            
            self._last_fingerprint = fingerprint
            self._sequence += 1
    
    async def create_snapshot(self) -> Recording:
        """Create a snapshot of the current recording without stopping."""
        if not self._running or not self._recording:
            raise RuntimeError("Recorder is not running")
        
        async with self._lock:
            # Create a copy with current state
            snapshot_metadata = RunMetadata.from_dict(self._metadata.to_dict())
            snapshot_metadata.duration_ms = (time.perf_counter() - self._start_time) * 1000
            
            return Recording(
                metadata=snapshot_metadata,
                initial_input=dict(self._recording.initial_input),
                events=list(self._recording.events),
            )


__all__ = [
    "RecordedEvent",
    "Recording",
    "ReplayRecorder",
]
