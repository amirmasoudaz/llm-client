"""
Action protocol for human-in-the-loop interactions.

This module provides the action system that enables agents to:
- Pause execution and request human input
- Persist action state for async resolution
- Resume execution after action is resolved
"""

from .types import (
    ActionStatus,
    ActionType,
    ActionRecord,
)
from .manager import (
    ActionManager,
    ActionSpec,
    ActionRequiredError,
)
from .store import (
    ActionStore,
    InMemoryActionStore,
    ActionFilter,
)

__all__ = [
    "ActionStatus",
    "ActionType",
    "ActionRecord",
    "ActionManager",
    "ActionSpec",
    "ActionRequiredError",
    "ActionStore",
    "InMemoryActionStore",
    "ActionFilter",
]
