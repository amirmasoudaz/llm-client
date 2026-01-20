"""
Request specification and context for deterministic execution.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from blake3 import blake3

from .providers.types import Message
from .serialization import stable_json_dumps
from .tools.base import Tool


@dataclass(frozen=True)
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    tenant_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "tags": dict(self.tags),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class RequestSpec:
    provider: str
    model: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[Union[str, Dict[str, Any], Type]] = None
    reasoning_effort: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        tools_payload = None
        if self.tools:
            tools_payload = sorted(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    }
                    for tool in self.tools
                ],
                key=lambda t: t["name"],
            )

        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "tools": tools_payload,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "reasoning_effort": self.reasoning_effort,
            "reasoning": self.reasoning,
            "extra": dict(self.extra),
            "stream": self.stream,
        }

    def cache_key(self) -> str:
        payload = self.to_dict()
        return blake3(stable_json_dumps(payload).encode("utf-8")).hexdigest()


__all__ = ["RequestContext", "RequestSpec"]
