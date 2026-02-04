"""
Run metadata for replay validation.

This module provides:
- RunMetadata: Captures version stamps for replay compatibility
- EventFingerprint: Cryptographic fingerprint for event validation
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class ReplayValidationError(Exception):
    """Raised when replay validation fails."""
    
    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class EventFingerprint:
    """Cryptographic fingerprint for validating event integrity.
    
    Fingerprints are computed from:
    - Event type
    - Event data (deterministic serialization)
    - Sequence number
    - Parent fingerprint (for chain validation)
    
    This allows validation that replay events match recorded events.
    """
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
        """Compute fingerprint for an event."""
        # Deterministic serialization
        canonical = json.dumps(
            {
                "type": event_type,
                "data": event_data,
                "seq": sequence,
                "parent": parent_hash,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        
        hash_value = hashlib.sha256(canonical.encode()).hexdigest()[:32]
        
        return cls(
            hash=hash_value,
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
        """Validate that event matches this fingerprint."""
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
    """Metadata for a recorded run, enabling replay validation.
    
    Captures:
    - Version stamps for all components
    - Configuration fingerprints
    - Timing information
    - Event chain hash for integrity
    
    Example:
        ```python
        metadata = RunMetadata.create(
            runtime_version="1.0.0",
            llm_client_version="0.5.0",
            model_version="gpt-4-0125",
        )
        
        # Later, validate replay compatibility
        metadata.validate_compatibility(current_metadata)
        ```
    """
    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str | None = None
    session_id: str | None = None
    
    # Version stamps
    runtime_version: str | None = None
    llm_client_version: str | None = None
    operator_version: str | None = None
    model_version: str | None = None
    
    # Configuration fingerprints
    config_hash: str | None = None  # Hash of runtime config
    policy_hash: str | None = None  # Hash of policy config
    tools_hash: str | None = None   # Hash of registered tools
    
    # Timing
    recorded_at: float = field(default_factory=time.time)
    duration_ms: float | None = None
    
    # Event chain
    event_count: int = 0
    first_event_hash: str | None = None
    last_event_hash: str | None = None
    
    # Model responses (for deterministic replay)
    model_response_hashes: list[str] = field(default_factory=list)
    
    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    
    # Schema version
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
        """Create metadata with computed hashes."""
        config_hash = None
        if config:
            config_json = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        policy_hash = None
        if policy:
            policy_json = json.dumps(policy, sort_keys=True)
            policy_hash = hashlib.sha256(policy_json.encode()).hexdigest()[:16]
        
        tools_hash = None
        if tools:
            tools_str = ",".join(sorted(tools))
            tools_hash = hashlib.sha256(tools_str.encode()).hexdigest()[:16]
        
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
        """Update event chain tracking."""
        self.event_count += 1
        if self.first_event_hash is None:
            self.first_event_hash = fingerprint.hash
        self.last_event_hash = fingerprint.hash
    
    def add_model_response(self, response_content: str) -> None:
        """Add a model response hash for replay validation."""
        response_hash = hashlib.sha256(response_content.encode()).hexdigest()[:16]
        self.model_response_hashes.append(response_hash)
    
    def validate_compatibility(
        self,
        other: RunMetadata,
        strict: bool = False,
    ) -> tuple[bool, list[str]]:
        """Validate compatibility with another metadata record.
        
        Args:
            other: Metadata to compare against
            strict: If True, require exact version match
        
        Returns:
            Tuple of (is_compatible, list of warnings/errors)
        """
        issues: list[str] = []
        
        # Version checks
        if strict:
            if self.runtime_version != other.runtime_version:
                issues.append(
                    f"Runtime version mismatch: {self.runtime_version} vs {other.runtime_version}"
                )
            if self.llm_client_version != other.llm_client_version:
                issues.append(
                    f"LLM client version mismatch: {self.llm_client_version} vs {other.llm_client_version}"
                )
            if self.model_version != other.model_version:
                issues.append(
                    f"Model version mismatch: {self.model_version} vs {other.model_version}"
                )
        else:
            # Non-strict: warn on major version differences
            if self.model_version != other.model_version:
                issues.append(
                    f"Warning: Model version differs: {self.model_version} vs {other.model_version}"
                )
        
        # Config checks
        if self.config_hash and other.config_hash and self.config_hash != other.config_hash:
            issues.append("Configuration hash mismatch")
        
        if self.policy_hash and other.policy_hash and self.policy_hash != other.policy_hash:
            issues.append("Policy hash mismatch")
        
        if self.tools_hash and other.tools_hash and self.tools_hash != other.tools_hash:
            issues.append("Tools hash mismatch")
        
        # Schema version
        if self.schema_version != other.schema_version:
            issues.append(
                f"Schema version mismatch: {self.schema_version} vs {other.schema_version}"
            )
        
        is_compatible = len(issues) == 0 or (not strict and all("Warning:" in i for i in issues))
        return is_compatible, issues
    
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
            "model_response_hashes": self.model_response_hashes,
            "tags": self.tags,
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
            model_response_hashes=data.get("model_response_hashes", []),
            tags=data.get("tags", {}),
            schema_version=data.get("schema_version", 1),
        )


__all__ = [
    "RunMetadata",
    "EventFingerprint",
    "ReplayValidationError",
]
