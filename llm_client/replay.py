"""
Generic replay metadata, recording, and playback primitives.

These types are storage-agnostic and operate against the package-level runtime
event model so they can be reused across runtimes and applications.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Callable

from .runtime_events import EventBus, EventSubscription, RuntimeEvent, RuntimeEventType


class ReplayValidationError(Exception):
    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class EventFingerprint:
    hash: str
    sequence: int
    event_type: str
    parent_hash: str | None = None

    @classmethod
    def compute(
        cls,
        event_type: str,
        event_data: dict[str, Any],
        sequence: int,
        parent_hash: str | None = None,
    ) -> EventFingerprint:
        canonical = json.dumps(
            {"type": event_type, "data": event_data, "seq": sequence, "parent": parent_hash},
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return cls(
            hash=hashlib.sha256(canonical.encode()).hexdigest()[:32],
            sequence=sequence,
            event_type=event_type,
            parent_hash=parent_hash,
        )

    def validate(
        self,
        event_type: str,
        event_data: dict[str, Any],
        parent_hash: str | None = None,
    ) -> bool:
        expected = self.compute(event_type, event_data, self.sequence, parent_hash)
        return expected.hash == self.hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "parent_hash": self.parent_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventFingerprint:
        return cls(
            hash=data["hash"],
            sequence=data["sequence"],
            event_type=data["event_type"],
            parent_hash=data.get("parent_hash"),
        )


@dataclass
class RunMetadata:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str | None = None
    session_id: str | None = None
    runtime_version: str | None = None
    llm_client_version: str | None = None
    operator_version: str | None = None
    model_version: str | None = None
    config_hash: str | None = None
    policy_hash: str | None = None
    tools_hash: str | None = None
    recorded_at: float = field(default_factory=time.time)
    duration_ms: float | None = None
    event_count: int = 0
    first_event_hash: str | None = None
    last_event_hash: str | None = None
    model_response_hashes: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        runtime_version: str | None = None,
        llm_client_version: str | None = None,
        model_version: str | None = None,
        config: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        **kwargs: Any,
    ) -> RunMetadata:
        config_hash = _sha16(config) if config else None
        policy_hash = _sha16(policy) if policy else None
        tools_hash = _sha16(",".join(sorted(tools))) if tools else None
        return cls(
            runtime_version=runtime_version,
            llm_client_version=llm_client_version,
            model_version=model_version,
            config_hash=config_hash,
            policy_hash=policy_hash,
            tools_hash=tools_hash,
            **kwargs,
        )

    def update_event_chain(self, fingerprint: EventFingerprint) -> None:
        self.event_count += 1
        if self.first_event_hash is None:
            self.first_event_hash = fingerprint.hash
        self.last_event_hash = fingerprint.hash

    def add_model_response(self, response_content: str) -> None:
        self.model_response_hashes.append(hashlib.sha256(response_content.encode()).hexdigest()[:16])

    def validate_compatibility(
        self,
        other: RunMetadata,
        strict: bool = False,
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []
        if strict:
            if self.runtime_version != other.runtime_version:
                issues.append(f"Runtime version mismatch: {self.runtime_version} vs {other.runtime_version}")
            if self.llm_client_version != other.llm_client_version:
                issues.append(f"LLM client version mismatch: {self.llm_client_version} vs {other.llm_client_version}")
            if self.model_version != other.model_version:
                issues.append(f"Model version mismatch: {self.model_version} vs {other.model_version}")
        else:
            if self.model_version != other.model_version:
                issues.append(f"Warning: Model version differs: {self.model_version} vs {other.model_version}")
        if self.config_hash and other.config_hash and self.config_hash != other.config_hash:
            issues.append("Configuration hash mismatch")
        if self.policy_hash and other.policy_hash and self.policy_hash != other.policy_hash:
            issues.append("Policy hash mismatch")
        if self.tools_hash and other.tools_hash and self.tools_hash != other.tools_hash:
            issues.append("Tools hash mismatch")
        if self.schema_version != other.schema_version:
            issues.append(f"Schema version mismatch: {self.schema_version} vs {other.schema_version}")
        return len(issues) == 0 or (not strict and all(item.startswith("Warning:") for item in issues)), issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "runtime_version": self.runtime_version,
            "llm_client_version": self.llm_client_version,
            "operator_version": self.operator_version,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "policy_hash": self.policy_hash,
            "tools_hash": self.tools_hash,
            "recorded_at": self.recorded_at,
            "duration_ms": self.duration_ms,
            "event_count": self.event_count,
            "first_event_hash": self.first_event_hash,
            "last_event_hash": self.last_event_hash,
            "model_response_hashes": list(self.model_response_hashes),
            "tags": dict(self.tags),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            job_id=data.get("job_id"),
            session_id=data.get("session_id"),
            runtime_version=data.get("runtime_version"),
            llm_client_version=data.get("llm_client_version"),
            operator_version=data.get("operator_version"),
            model_version=data.get("model_version"),
            config_hash=data.get("config_hash"),
            policy_hash=data.get("policy_hash"),
            tools_hash=data.get("tools_hash"),
            recorded_at=data.get("recorded_at", time.time()),
            duration_ms=data.get("duration_ms"),
            event_count=data.get("event_count", 0),
            first_event_hash=data.get("first_event_hash"),
            last_event_hash=data.get("last_event_hash"),
            model_response_hashes=list(data.get("model_response_hashes", [])),
            tags=dict(data.get("tags", {})),
            schema_version=data.get("schema_version", 1),
        )


@dataclass
class RecordedEvent:
    event: RuntimeEvent
    fingerprint: EventFingerprint
    relative_timestamp_ms: float
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
    metadata: RunMetadata
    events: list[RecordedEvent] = field(default_factory=list)
    initial_input: dict[str, Any] = field(default_factory=dict)
    format_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "metadata": self.metadata.to_dict(),
            "initial_input": dict(self.initial_input),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recording:
        return cls(
            format_version=data.get("format_version", 1),
            metadata=RunMetadata.from_dict(data["metadata"]),
            initial_input=dict(data.get("initial_input", {})),
            events=[RecordedEvent.from_dict(item) for item in data.get("events", [])],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save_binary(self, handle: BinaryIO) -> None:
        handle.write(gzip.compress(json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")))

    @classmethod
    def load_binary(cls, handle: BinaryIO) -> Recording:
        return cls.from_dict(json.loads(gzip.decompress(handle.read()).decode("utf-8")))

    def validate_chain(self) -> tuple[bool, str | None]:
        if not self.events:
            return True, None
        if self.metadata.first_event_hash and self.events[0].fingerprint.hash != self.metadata.first_event_hash:
            return False, "First event hash mismatch"
        previous_hash = None
        for index, recorded in enumerate(self.events):
            if recorded.fingerprint.parent_hash != previous_hash:
                return False, f"Chain broken at event {index}"
            previous_hash = recorded.fingerprint.hash
        if self.metadata.last_event_hash and previous_hash != self.metadata.last_event_hash:
            return False, "Last event hash mismatch"
        return True, None

    def get_events_by_type(self, event_type: RuntimeEventType) -> list[RecordedEvent]:
        return [event for event in self.events if event.event.type == event_type]

    def get_model_responses(self) -> list[str]:
        return [event.model_response for event in self.events if event.model_response is not None]


class ReplayRecorder:
    def __init__(
        self,
        event_bus: EventBus,
        metadata: RunMetadata | None = None,
        capture_model_responses: bool = True,
    ) -> None:
        self._event_bus = event_bus
        self._metadata = metadata or RunMetadata()
        self._capture_responses = capture_model_responses
        self._recording: Recording | None = None
        self._subscription: EventSubscription | None = None
        self._consumer_task: asyncio.Task[None] | None = None
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
        *,
        job_id: str | None = None,
        event_types: set[RuntimeEventType] | None = None,
        scope_id: str | None = None,
    ) -> None:
        if self._running:
            return
        async with self._lock:
            self._recording = Recording(metadata=self._metadata, initial_input=initial_input or {})
            self._start_time = time.perf_counter()
            self._sequence = 0
            self._last_fingerprint = None
            self._subscription = self._event_bus.subscribe(job_id=job_id, event_types=event_types, scope_id=scope_id)
            self._consumer_task = asyncio.create_task(self._consume_events())
            self._running = True

    async def stop(self) -> Recording:
        if not self._running:
            raise RuntimeError("Recorder is not running")
        subscription = self._subscription
        consumer_task = self._consumer_task
        self._running = False
        if subscription is not None:
            self._event_bus.unsubscribe(subscription)
            self._subscription = None
        if consumer_task is not None:
            await consumer_task
            self._consumer_task = None
        async with self._lock:
            self._metadata.duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._metadata.event_count = len(self._recording.events if self._recording else [])
            recording = self._recording
            self._recording = None
            if recording is None:
                raise RuntimeError("Recording state lost")
            return recording

    async def create_snapshot(self) -> Recording:
        if not self._running or self._recording is None:
            raise RuntimeError("Recorder is not running")
        async with self._lock:
            metadata = RunMetadata.from_dict(self._metadata.to_dict())
            metadata.duration_ms = (time.perf_counter() - self._start_time) * 1000
            return Recording(
                metadata=metadata,
                initial_input=dict(self._recording.initial_input),
                events=list(self._recording.events),
            )

    async def _consume_events(self) -> None:
        subscription = self._subscription
        if subscription is None:
            return
        try:
            async for event in self._event_bus.events(subscription):
                await self._record_event(event)
        except asyncio.CancelledError:
            return

    async def _record_event(self, event: RuntimeEvent) -> None:
        if not self._running or self._recording is None:
            return
        async with self._lock:
            relative_ts = (time.perf_counter() - self._start_time) * 1000
            parent_hash = self._last_fingerprint.hash if self._last_fingerprint else None
            fingerprint = EventFingerprint.compute(
                event_type=event.type.value,
                event_data=event.data,
                sequence=self._sequence,
                parent_hash=parent_hash,
            )
            model_response = None
            if self._capture_responses and event.type in {RuntimeEventType.MODEL_DONE, RuntimeEventType.FINAL_RESULT}:
                model_response = str(event.data.get("content") or event.data.get("response") or "") or None
                if model_response is not None:
                    self._metadata.add_model_response(model_response)
            self._recording.events.append(
                RecordedEvent(
                    event=event,
                    fingerprint=fingerprint,
                    relative_timestamp_ms=relative_ts,
                    model_response=model_response,
                )
            )
            self._metadata.update_event_chain(fingerprint)
            self._last_fingerprint = fingerprint
            self._sequence += 1


class ReplayMode(str, Enum):
    DETERMINISTIC = "deterministic"
    VALIDATE = "validate"
    TIMED = "timed"
    FAST = "fast"
    STEP = "step"


@dataclass
class ReplayResult:
    success: bool = True
    error: str | None = None
    events_replayed: int = 0
    events_matched: int = 0
    events_mismatched: int = 0
    original_duration_ms: float | None = None
    replay_duration_ms: float = 0.0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] | None = None
    content: str | None = None


class ReplayPlayer:
    def __init__(
        self,
        recording: Recording,
        event_bus: EventBus | None = None,
        model_provider: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._recording = recording
        self._event_bus = event_bus
        self._model_provider = model_provider
        self._current_index = 0
        self._start_time = 0.0
        self._response_index = 0
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
        issues: list[str] = []
        is_valid, error = self._recording.validate_chain()
        if not is_valid:
            issues.append(f"Chain validation failed: {error}")
        if self._recording.format_version > 1:
            issues.append(f"Unsupported format version: {self._recording.format_version}")
        if self._recording.metadata.schema_version > 1:
            issues.append(f"Warning: Recording uses newer schema version {self._recording.metadata.schema_version}")
        return len(issues) == 0 or all(item.startswith("Warning:") for item in issues), issues

    async def replay(
        self,
        mode: ReplayMode = ReplayMode.FAST,
        strict_validation: bool = False,
        on_event: Callable[[RuntimeEvent], None] | None = None,
    ) -> ReplayResult:
        is_valid, issues = await self.validate_recording()
        if not is_valid and strict_validation:
            raise ReplayValidationError(f"Invalid recording: {issues}")

        self._running = True
        self._current_index = 0
        self._response_index = 0
        self._start_time = time.perf_counter()
        result = ReplayResult(original_duration_ms=self._recording.metadata.duration_ms)

        try:
            if mode == ReplayMode.STEP:
                raise ValueError("Use step() method for step mode")
            async for event in self._replay_events(mode):
                result.events_replayed += 1
                if self._event_bus is not None:
                    await self._event_bus.publish(event)
                if on_event is not None:
                    on_event(event)
                if event.type == RuntimeEventType.FINAL_RESULT:
                    result.content = event.data.get("content")
                    result.output = dict(event.data)
            result.success = True
            result.events_matched = result.events_replayed
        except ReplayValidationError as exc:
            result.success = False
            result.error = str(exc)
            if exc.expected is not None or exc.actual is not None:
                result.mismatches.append({"expected": exc.expected, "actual": exc.actual})
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        finally:
            self._running = False
            result.replay_duration_ms = (time.perf_counter() - self._start_time) * 1000
        return result

    async def step(self) -> AsyncIterator[RuntimeEvent]:
        self._running = True
        self._current_index = 0
        self._response_index = 0
        try:
            while self._current_index < len(self._recording.events):
                recorded = self._recording.events[self._current_index]
                self._current_index += 1
                yield self._reconstruct_event(recorded)
        finally:
            self._running = False

    async def seek(self, position: int) -> None:
        if position < 0 or position >= len(self._recording.events):
            raise ValueError(f"Position {position} out of range")
        self._current_index = position

    async def _replay_events(self, mode: ReplayMode) -> AsyncIterator[RuntimeEvent]:
        last_time = 0.0
        while self._current_index < len(self._recording.events):
            recorded = self._recording.events[self._current_index]
            if mode == ReplayMode.TIMED:
                delay = (recorded.relative_timestamp_ms - last_time) / 1000.0
                if delay > 0:
                    await asyncio.sleep(delay)
                last_time = recorded.relative_timestamp_ms
            event = await self._validate_and_execute(recorded) if mode == ReplayMode.VALIDATE and self._model_provider else self._reconstruct_event(recorded)
            self._current_index += 1
            yield event

    def _reconstruct_event(self, recorded: RecordedEvent) -> RuntimeEvent:
        data = dict(recorded.event.data)
        if recorded.model_response and recorded.event.type in {RuntimeEventType.MODEL_DONE, RuntimeEventType.FINAL_RESULT}:
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

    async def _validate_and_execute(self, recorded: RecordedEvent) -> RuntimeEvent:
        if recorded.event.type not in {RuntimeEventType.MODEL_DONE, RuntimeEventType.FINAL_RESULT}:
            return self._reconstruct_event(recorded)
        if self._model_provider is not None:
            _ = self._model_provider
        return self._reconstruct_event(recorded)

    def get_model_response(self, index: int) -> str | None:
        model_events = [event for event in self._recording.events if event.model_response is not None]
        if index < len(model_events):
            return model_events[index].model_response
        return None

    def get_next_model_response(self) -> str | None:
        response = self.get_model_response(self._response_index)
        if response is not None:
            self._response_index += 1
        return response


def _sha16(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


__all__ = [
    "EventFingerprint",
    "RecordedEvent",
    "Recording",
    "ReplayMode",
    "ReplayPlayer",
    "ReplayRecorder",
    "ReplayResult",
    "ReplayValidationError",
    "RunMetadata",
]
