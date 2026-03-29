from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_client.providers.types import CompletionResult, EmbeddingResult, StreamEvent, Usage


@dataclass(frozen=True)
class FakeModel:
    key: str = "gpt-5-mini"
    model_name: str = "gpt-5-mini"

    @staticmethod
    def count_tokens(content: Any) -> int:
        return len(str(content or "").split())

    @staticmethod
    def parse_usage(raw_usage: dict[str, Any]) -> dict[str, Any]:
        return dict(raw_usage)


class ScriptedProvider:
    def __init__(
        self,
        *,
        complete_script: list[Any] | None = None,
        stream_script: list[Any] | None = None,
        embed_script: list[Any] | None = None,
        model_name: str = "gpt-5-mini",
    ) -> None:
        self.model = FakeModel(key=model_name, model_name=model_name)
        self.model_name = model_name
        self._complete_script = list(complete_script or [])
        self._stream_script = list(stream_script or [])
        self._embed_script = list(embed_script or [])
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    async def complete(self, messages, **kwargs):
        self.complete_calls.append({"messages": messages, "kwargs": kwargs})
        if not self._complete_script:
            raise AssertionError("Unexpected complete() call")
        item = self._complete_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        if not self._stream_script:
            raise AssertionError("Unexpected stream() call")
        item = self._stream_script.pop(0)
        if isinstance(item, Exception):
            raise item
        for event in item:
            yield event

    async def embed(self, inputs, **kwargs):
        self.embed_calls.append({"inputs": inputs, "kwargs": kwargs})
        if self._embed_script:
            item = self._embed_script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        vectors = [[0.0, 1.0, 0.5]]
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        return EmbeddingResult(
            embeddings=vectors * len(texts),
            usage=Usage(input_tokens=len(texts), total_tokens=len(texts)),
            model=self.model_name,
            status=200,
        )

    def count_tokens(self, content: Any) -> int:
        return self.model.count_tokens(content)

    def parse_usage(self, raw_usage: dict[str, Any]) -> Usage:
        parsed = self.model.parse_usage(raw_usage)
        return Usage(
            input_tokens=int(parsed.get("input_tokens", 0) or 0),
            output_tokens=int(parsed.get("output_tokens", 0) or 0),
            total_tokens=int(parsed.get("total_tokens", 0) or 0),
        )

    async def close(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def ok_result(content: str, *, model: str = "gpt-5-mini") -> CompletionResult:
    return CompletionResult(content=content, usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2), model=model, status=200)


def error_result(status: int, message: str, *, model: str = "gpt-5-mini") -> CompletionResult:
    return CompletionResult(content=None, usage=Usage(), model=model, status=status, error=message)


__all__ = [
    "FakeModel",
    "ScriptedProvider",
    "error_result",
    "ok_result",
]
