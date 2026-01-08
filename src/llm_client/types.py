from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMRequest:
    model: str
    category: Literal["completions", "embeddings", "responses"]
    messages: Any | None = None
    input: Any | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    response_format: Any | None = None
    reasoning: dict | None = None
    reasoning_effort: str | None = None
    stream: bool = False
    stream_options: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResult:
    output: Any | None
    usage: dict
    status: int
    error: str
    params: dict
    tool_calls: list[dict] | None = None
    message: dict | None = None
    raw: Any | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "params": self.params,
            "output": self.output,
            "usage": self.usage,
            "status": self.status,
            "error": self.error,
        }
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.message is not None:
            data["message"] = self.message
        if self.metadata:
            data.update(self.metadata)
        return data


@dataclass
class LLMEvent:
    type: str
    data: dict


__all__ = ["LLMRequest", "LLMResult", "LLMEvent"]
