"""
Action types for human-in-the-loop protocol.

This module defines the ActionRecord and related types that form
the core of the action/pause/resume system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionStatus(str, Enum):
    """Action lifecycle states."""
    PENDING = "pending"      # Waiting for resolution
    RESOLVED = "resolved"    # Successfully resolved
    EXPIRED = "expired"      # Deadline passed
    CANCELLED = "cancelled"  # Explicitly cancelled

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {
            ActionStatus.RESOLVED,
            ActionStatus.EXPIRED,
            ActionStatus.CANCELLED,
        }


class ActionType(str, Enum):
    """Common action types.
    
    These are the standard action types that the runtime understands.
    Applications can use custom string types as well.
    """
    CONFIRM = "confirm"                    # User must confirm an action
    CHOOSE = "choose"                      # User must select from options
    INPUT = "input"                        # User must provide input
    UPLOAD = "upload"                      # User must upload a file
    REAUTH = "reauth"                      # User must re-authenticate
    APPLY_CHANGES = "apply_changes"        # User must approve changes
    APPROVAL = "approval"                  # General approval request
    CUSTOM = "custom"                      # Custom action type


@dataclass
class ActionRecord:
    """Persistent record of an action request.
    
    An action is a request from the runtime to an external system
    (usually a human user or UI) to do something and provide a result.
    
    The action protocol:
    1. Runtime creates action with PENDING status
    2. Job transitions to WAITING_ACTION
    3. External system receives action via event
    4. External system resolves action with resolution payload
    5. Job transitions back to RUNNING and continues
    """
    # Identity
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""  # Required - the job this action belongs to
    
    # Type and payload
    type: str = ActionType.CONFIRM.value  # Action type (string for flexibility)
    payload: dict[str, Any] = field(default_factory=dict)  # UI instructions
    
    # Status
    status: ActionStatus = ActionStatus.PENDING
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # Absolute timestamp
    resolved_at: float | None = None
    
    # Resolution
    resolution: dict[str, Any] | None = None
    resolution_error: str | None = None
    
    # Resume token for security
    resume_token: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Schema version
    schema_version: int = 1

    @property
    def is_expired(self) -> bool:
        """Check if the action has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def resolve(self, resolution: dict[str, Any]) -> ActionRecord:
        """Create a resolved copy of this action."""
        return ActionRecord(
            action_id=self.action_id,
            job_id=self.job_id,
            type=self.type,
            payload=dict(self.payload),
            status=ActionStatus.RESOLVED,
            created_at=self.created_at,
            expires_at=self.expires_at,
            resolved_at=time.time(),
            resolution=resolution,
            resolution_error=None,
            resume_token=self.resume_token,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )

    def expire(self) -> ActionRecord:
        """Create an expired copy of this action."""
        return ActionRecord(
            action_id=self.action_id,
            job_id=self.job_id,
            type=self.type,
            payload=dict(self.payload),
            status=ActionStatus.EXPIRED,
            created_at=self.created_at,
            expires_at=self.expires_at,
            resolved_at=time.time(),
            resolution=None,
            resolution_error="Action expired",
            resume_token=self.resume_token,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )

    def cancel(self, reason: str | None = None) -> ActionRecord:
        """Create a cancelled copy of this action."""
        return ActionRecord(
            action_id=self.action_id,
            job_id=self.job_id,
            type=self.type,
            payload=dict(self.payload),
            status=ActionStatus.CANCELLED,
            created_at=self.created_at,
            expires_at=self.expires_at,
            resolved_at=time.time(),
            resolution=None,
            resolution_error=reason or "Action cancelled",
            resume_token=self.resume_token,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_id": self.action_id,
            "job_id": self.job_id,
            "type": self.type,
            "payload": dict(self.payload),
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "resolved_at": self.resolved_at,
            "resolution": self.resolution,
            "resolution_error": self.resolution_error,
            "resume_token": self.resume_token,
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRecord:
        """Deserialize from dictionary."""
        return cls(
            action_id=data.get("action_id", str(uuid.uuid4())),
            job_id=data.get("job_id", ""),
            type=data.get("type", ActionType.CONFIRM.value),
            payload=dict(data.get("payload", {})),
            status=ActionStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            resolved_at=data.get("resolved_at"),
            resolution=data.get("resolution"),
            resolution_error=data.get("resolution_error"),
            resume_token=data.get("resume_token", str(uuid.uuid4())),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", 1),
        )

    def to_event_payload(self) -> dict[str, Any]:
        """Convert to event payload (excludes sensitive fields)."""
        return {
            "action_id": self.action_id,
            "job_id": self.job_id,
            "type": self.type,
            "payload": self.payload,
            "expires_at": self.expires_at,
            "resume_token": self.resume_token,
        }


__all__ = [
    "ActionStatus",
    "ActionType",
    "ActionRecord",
]
