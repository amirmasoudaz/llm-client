"""
Generic structured tool-loop runtime for LLM completions.

This module contains reusable runtime pieces for:
- normalizing model-emitted tool calls
- executing tool calls with timeout and schema validation
- feeding tool results back into a multi-turn LLM loop
- optional engine-backed execution and streaming token extraction
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from ..content import ensure_completion_result
from ..errors import normalize_tool_failure
from ..providers.types import CompletionResult, StreamEventType, Usage
from ..request_builders import build_content_request_envelope

_ASSISTANT_TEXT_ANCHOR = re.compile(r'"assistant_message"\s*:\s*\{\s*"text"\s*:\s*"', re.IGNORECASE)
_STREAM_INTERNAL_ENDPOINT_RE = re.compile(r"/v1/threads/\d+/context", re.IGNORECASE)
_STREAM_THREAD_NUMBER_RE = re.compile(r"\bthread\s+#?\d+\b", re.IGNORECASE)
_STREAM_INTERNAL_FIELD_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfunding_request_id\b", re.IGNORECASE), "funding request"),
    (re.compile(r"\bprofessor_id\b", re.IGNORECASE), "professor"),
    (re.compile(r"\bthread_id\b", re.IGNORECASE), "thread"),
    (re.compile(r"\bIDs\b", re.IGNORECASE), "details"),
    (re.compile(r"\bID\b", re.IGNORECASE), "detail"),
)


@dataclass(frozen=True)
class StructuredToolRuntime:
    tools_by_name: dict[str, Any]
    provider_tools: list[Any]
    max_tool_calls: int
    max_tool_call_depth: int


@dataclass(frozen=True)
class StructuredToolLoopError:
    code: str
    message: str
    category: str = "validation"
    retryable: bool = False
    details: dict[str, Any] | None = None

    def to_normalized_failure(self, *, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        return normalize_tool_failure(
            self,
            provider=provider,
            model=model,
            operation="tools",
        ).to_dict()


class RuntimeToolError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        category: str = "validation",
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.category = category
        self.retryable = retryable
        self.details = details or {}


@dataclass
class _ToolExecBatch:
    messages: list[dict[str, Any]]
    error: StructuredToolLoopError | None = None


async def complete_with_tools(
    *,
    engine: Any | None,
    provider: Any,
    messages: list[dict[str, Any]],
    completion_kwargs: dict[str, Any],
    runtime: StructuredToolRuntime,
    tool_timeout_ms: int,
    soft_limit: int = 5,
    token_delta_callback: Callable[[str], Awaitable[None]] | None = None,
    token_delta_mode: str | None = None,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[CompletionResult | None, StructuredToolLoopError | None, list[dict[str, Any]]]:
    if not runtime.provider_tools:
        completion = await _complete_provider_turn(
            engine=engine,
            provider=provider,
            messages=messages,
            request_kwargs=completion_kwargs,
            token_delta_callback=token_delta_callback,
            token_delta_mode=token_delta_mode,
        )
        completion_tool_calls, tool_call_error = normalize_completion_tool_calls(getattr(completion, "tool_calls", None))
        if tool_call_error is not None:
            return None, StructuredToolLoopError(
                code="tool_call_invalid",
                message=tool_call_error,
                category="validation",
                retryable=False,
            ), list(messages)
        if completion_tool_calls:
            return None, StructuredToolLoopError(
                code="tool_not_allowed",
                message="llm requested tools but no tool runtime was declared",
                category="policy_denied",
                retryable=False,
            ), list(messages)
        return completion, None, list(messages)

    working_messages = list(messages)
    tool_calls_used = 0
    tool_call_depth = 0

    while True:
        request_kwargs = dict(completion_kwargs)
        request_kwargs["tools"] = runtime.provider_tools
        request_kwargs["tool_choice"] = "auto"
        completion = await _complete_provider_turn(
            engine=engine,
            provider=provider,
            messages=working_messages,
            request_kwargs=request_kwargs,
            token_delta_callback=token_delta_callback,
            token_delta_mode=token_delta_mode,
        )
        completion_tool_calls, tool_call_error = normalize_completion_tool_calls(getattr(completion, "tool_calls", None))
        if tool_call_error is not None:
            return None, StructuredToolLoopError(
                code="tool_call_invalid",
                message=tool_call_error,
                category="validation",
                retryable=False,
            ), working_messages
        if not completion_tool_calls:
            return completion, None, working_messages

        tool_call_depth += 1
        if tool_call_depth == soft_limit:
            working_messages.append(
                {
                    "role": "system",
                    "content": (
                        "You have used many tool calls. Please finalize your response to the user now. "
                        "If you need more information, ask the user in your response."
                    ),
                }
            )
        if tool_call_depth > runtime.max_tool_call_depth:
            return None, StructuredToolLoopError(
                code="tool_call_depth_exceeded",
                message=f"llm tool call depth exceeded: {runtime.max_tool_call_depth}",
                category="rate_limited",
                retryable=False,
            ), working_messages

        assistant_tool_message = build_assistant_tool_call_message(
            content=getattr(completion, "content", None),
            tool_calls=completion_tool_calls,
        )
        if assistant_tool_message is not None:
            working_messages.append(assistant_tool_message)

        for tool_call in completion_tool_calls:
            tool_name = str(tool_call.get("name") or "").strip()
            if tool_name not in runtime.tools_by_name:
                return None, StructuredToolLoopError(
                    code="tool_not_allowed",
                    message=f"llm requested undeclared tool: {tool_name}",
                    category="policy_denied",
                    retryable=False,
                ), working_messages
            if tool_calls_used + len(completion_tool_calls) > runtime.max_tool_calls:
                return None, StructuredToolLoopError(
                    code="tool_call_limit_exceeded",
                    message=f"llm tool call limit exceeded: {runtime.max_tool_calls}",
                    category="rate_limited",
                    retryable=False,
                ), working_messages

        tool_exec_result = (
            await _execute_tools_parallel(
                runtime=runtime,
                tool_calls=completion_tool_calls,
                tool_timeout_ms=tool_timeout_ms,
            )
            if len(completion_tool_calls) > 1
            else await _execute_tools_sequential(
                runtime=runtime,
                tool_calls=completion_tool_calls,
                tool_timeout_ms=tool_timeout_ms,
            )
        )

        if tool_exec_result.error is not None:
            return None, tool_exec_result.error, working_messages

        tool_calls_used += len(completion_tool_calls)
        working_messages.extend(tool_exec_result.messages)

        if progress_callback is not None:
            try:
                await progress_callback(f"Processed {len(completion_tool_calls)} tool call(s).")
            except Exception:
                pass


async def _execute_tools_sequential(
    *,
    runtime: StructuredToolRuntime,
    tool_calls: list[dict[str, str]],
    tool_timeout_ms: int,
) -> _ToolExecBatch:
    messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        tool_call_id = str(tool_call.get("id") or "").strip() or str(uuid.uuid4())
        tool = runtime.tools_by_name[tool_name]
        try:
            tool_result = await execute_runtime_tool(
                tool,
                tool_call.get("arguments"),
                timeout_ms=tool_timeout_ms,
            )
        except RuntimeToolError as exc:
            details = dict(exc.details)
            details.setdefault("tool_name", tool_name)
            return _ToolExecBatch(
                messages=messages,
                error=StructuredToolLoopError(
                    code=exc.code,
                    message=exc.message,
                    category=exc.category,
                    retryable=exc.retryable,
                    details=details or None,
                ),
            )
        except Exception as exc:
            return _ToolExecBatch(
                messages=messages,
                error=StructuredToolLoopError(
                    code="tool_execution_error",
                    message=str(exc),
                    category="dependency",
                    retryable=False,
                    details={"tool_name": tool_name},
                ),
            )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result_to_content(tool_result),
            }
        )
    return _ToolExecBatch(messages=messages)


async def _execute_tools_parallel(
    *,
    runtime: StructuredToolRuntime,
    tool_calls: list[dict[str, str]],
    tool_timeout_ms: int,
) -> _ToolExecBatch:
    async def _run_one(tc: dict[str, str]) -> dict[str, Any] | StructuredToolLoopError:
        tool_name = str(tc.get("name") or "").strip()
        tool_call_id = str(tc.get("id") or "").strip() or str(uuid.uuid4())
        tool = runtime.tools_by_name[tool_name]
        try:
            result = await execute_runtime_tool(tool, tc.get("arguments"), timeout_ms=tool_timeout_ms)
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result_to_content(result),
            }
        except RuntimeToolError as exc:
            details = dict(exc.details)
            details.setdefault("tool_name", tool_name)
            return StructuredToolLoopError(
                code=exc.code,
                message=exc.message,
                category=exc.category,
                retryable=exc.retryable,
                details=details or None,
            )
        except Exception as exc:
            return StructuredToolLoopError(
                code="tool_execution_error",
                message=str(exc),
                category="dependency",
                retryable=False,
                details={"tool_name": tool_name},
            )

    results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
    messages: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, StructuredToolLoopError):
            return _ToolExecBatch(messages=messages, error=result)
        messages.append(result)
    return _ToolExecBatch(messages=messages)


async def _complete_provider_turn(
    *,
    engine: Any | None,
    provider: Any,
    messages: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    token_delta_callback: Callable[[str], Awaitable[None]] | None,
    token_delta_mode: str | None,
) -> CompletionResult:
    stream_fn = getattr(provider, "stream", None) if provider is not None else None
    if engine is None and (not callable(stream_fn) or token_delta_callback is None):
        return await provider.complete(messages, **request_kwargs)
    if engine is not None and token_delta_callback is None:
        return (
            await engine.complete_content(
                build_content_request_envelope(
                    engine=engine,
                    provider=provider,
                    messages=messages,
                    request_kwargs=request_kwargs,
                )
            )
        ).to_completion_result()

    extractor = (
        _AssistantMessageTextJSONExtractor()
        if str(token_delta_mode or "").strip().lower() == "conversation_assistant_text"
        else None
    )
    final_completion: CompletionResult | None = None
    error_status: int | None = None
    error_message: str | None = None

    event_stream = (
        engine.stream_content(
            build_content_request_envelope(
                engine=engine,
                provider=provider,
                messages=messages,
                request_kwargs=request_kwargs,
                stream=True,
            )
        )
        if engine is not None
        else stream_fn(messages, **request_kwargs)
    )

    async for event in event_stream:
        event_type = getattr(event, "type", None)
        event_data = getattr(event, "data", None)
        if event_type == StreamEventType.TOKEN:
            token_text = str(event_data or "")
            if extractor is not None and token_text:
                for safe_chunk in extractor.feed(token_text):
                    if safe_chunk:
                        await token_delta_callback(safe_chunk)
            continue
        if event_type == StreamEventType.DONE:
            try:
                final_completion = ensure_completion_result(event_data)
            except TypeError:
                final_completion = CompletionResult(
                    content=str(event_data or "") if event_data is not None else None,
                    status=200,
                )
            break
        if event_type == StreamEventType.ERROR:
            if isinstance(event_data, dict):
                try:
                    error_status = int(event_data.get("status") or 500)
                except Exception:
                    error_status = 500
                error_message = str(event_data.get("error") or "stream_error")
            else:
                error_status = 500
                error_message = str(event_data or "stream_error")
            break

    if final_completion is not None:
        return final_completion

    return CompletionResult(
        content=None,
        tool_calls=None,
        usage=Usage(),
        status=error_status or 500,
        error=error_message or "llm stream produced no terminal result",
    )
class _AssistantMessageTextJSONExtractor:
    def __init__(self) -> None:
        self._buffer = ""
        self._raw_text_start: int | None = None
        self._emitted_decoded_len = 0
        self._emitted_sanitized_len = 0
        self._closed = False
        self._stream_holdback_chars = 48

    def feed(self, token: str) -> list[str]:
        if self._closed:
            return []
        self._buffer += str(token or "")
        if self._raw_text_start is None:
            match = _ASSISTANT_TEXT_ANCHOR.search(self._buffer)
            if match is None:
                return []
            self._raw_text_start = match.end()
        assert self._raw_text_start is not None
        raw_fragment, closed = _extract_json_string_raw_fragment(self._buffer, self._raw_text_start)
        if raw_fragment is None:
            return []
        try:
            decoded = json.loads(f'"{raw_fragment}"')
        except Exception:
            return []
        if not isinstance(decoded, str):
            return []
        if closed:
            self._closed = True
        if len(decoded) <= self._emitted_decoded_len:
            return []
        self._emitted_decoded_len = len(decoded)
        sanitized = _sanitize_assistant_text_for_stream(decoded)
        emit_upto = len(sanitized) if self._closed else max(0, len(sanitized) - self._stream_holdback_chars)
        if emit_upto <= self._emitted_sanitized_len:
            return []
        delta = sanitized[self._emitted_sanitized_len : emit_upto]
        self._emitted_sanitized_len = emit_upto
        return [delta] if delta else []


def _sanitize_assistant_text_for_stream(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = _STREAM_INTERNAL_ENDPOINT_RE.sub("the context list", value)
    value = _STREAM_THREAD_NUMBER_RE.sub("this thread", value)
    for pattern, replacement in _STREAM_INTERNAL_FIELD_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    return value


def _extract_json_string_raw_fragment(buffer: str, raw_start: int) -> tuple[str | None, bool]:
    if raw_start < 0 or raw_start > len(buffer):
        return None, False
    chars: list[str] = []
    i = raw_start
    escaping = False
    unicode_remaining = 0
    while i < len(buffer):
        ch = buffer[i]
        if unicode_remaining > 0:
            if ch.lower() not in "0123456789abcdef":
                return "".join(chars[:-2]) if len(chars) >= 2 else "", False
            chars.append(ch)
            unicode_remaining -= 1
            i += 1
            continue
        if escaping:
            chars.append(ch)
            if ch == "u":
                unicode_remaining = 4
            escaping = False
            i += 1
            continue
        if ch == "\\":
            chars.append(ch)
            escaping = True
            i += 1
            continue
        if ch == '"':
            return "".join(chars), True
        chars.append(ch)
        i += 1
    if escaping and chars and chars[-1] == "\\":
        chars = chars[:-1]
    if unicode_remaining > 0:
        trim = min(len(chars), 2 + (4 - unicode_remaining))
        chars = chars[:-trim] if trim else chars
    return "".join(chars), False


def normalize_completion_tool_calls(value: Any) -> tuple[list[dict[str, str]], str | None]:
    if not isinstance(value, list):
        return [], None
    out: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            tool_id = str(item.get("id") or "").strip() or f"tool_call_{index}"
            tool_name = str(item.get("name") or "").strip()
            arguments = str(item.get("arguments") or "").strip()
            malformed = [key for key in ("name", "arguments") if key not in item]
        else:
            tool_id = str(getattr(item, "id", "") or "").strip() or f"tool_call_{index}"
            tool_name = str(getattr(item, "name", "") or "").strip()
            arguments = str(getattr(item, "arguments", "") or "").strip()
            malformed = []
        if malformed:
            return [], f"tool call missing required fields: {','.join(malformed)}"
        if not tool_name:
            return [], "tool call name must be non-empty"
        out.append({"id": tool_id, "name": tool_name, "arguments": arguments})
    return out, None


def build_assistant_tool_call_message(*, content: Any, tool_calls: list[dict[str, str]]) -> dict[str, Any] | None:
    if not tool_calls:
        return None
    return {
        "role": "assistant",
        "content": str(content or ""),
        "tool_calls": [
            {
                "id": item["id"],
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            }
            for item in tool_calls
        ],
    }


def tool_to_provider_definition(tool: Any) -> Any | None:
    to_openai = getattr(tool, "to_openai_format", None)
    if callable(to_openai):
        try:
            return to_openai()
        except Exception:
            return None
    if isinstance(tool, dict):
        return dict(tool)
    return None


async def execute_runtime_tool(tool: Any, arguments: Any, *, timeout_ms: int) -> Any:
    parsed_arguments = parse_tool_arguments(arguments)
    validation_errors = validate_runtime_tool_arguments(tool, parsed_arguments)
    if validation_errors:
        raise RuntimeToolError(
            code="tool_arguments_invalid",
            message="tool arguments failed schema validation",
            category="validation",
            retryable=False,
            details={"validation_errors": validation_errors},
        )

    execute = getattr(tool, "execute", None)
    if callable(execute):
        result = execute(**parsed_arguments)
        timeout_seconds = max(0.1, float(timeout_ms) / 1000.0)
        try:
            if inspect.isawaitable(result):
                return await asyncio.wait_for(result, timeout=timeout_seconds)
            return await asyncio.wait_for(
                asyncio.to_thread(_invoke_sync_callable, execute, parsed_arguments),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeToolError(
                code="tool_call_timeout",
                message=f"tool call exceeded timeout ({timeout_ms}ms)",
                category="timeout",
                retryable=False,
            ) from exc

    execute_json = getattr(tool, "execute_json", None)
    if callable(execute_json):
        arguments_json = json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":"))
        result = execute_json(arguments_json)
        timeout_seconds = max(0.1, float(timeout_ms) / 1000.0)
        try:
            if inspect.isawaitable(result):
                return await asyncio.wait_for(result, timeout=timeout_seconds)
            return await asyncio.wait_for(
                asyncio.to_thread(_invoke_sync_json_callable, execute_json, arguments_json),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeToolError(
                code="tool_call_timeout",
                message=f"tool call exceeded timeout ({timeout_ms}ms)",
                category="timeout",
                retryable=False,
            ) from exc

    raise RuntimeToolError(
        code="tool_runtime_not_executable",
        message="runtime tool is not executable",
        category="operator_bug",
        retryable=False,
    )


def _invoke_sync_callable(fn: Any, parsed_arguments: dict[str, Any]) -> Any:
    return fn(**parsed_arguments)


def _invoke_sync_json_callable(fn: Any, arguments_json: str) -> Any:
    return fn(arguments_json)


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    arguments_text = str(arguments or "").strip()
    if not arguments_text:
        return {}
    try:
        parsed = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise RuntimeToolError(
            code="tool_arguments_invalid_json",
            message=f"tool arguments are not valid JSON: {exc}",
            category="validation",
            retryable=False,
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeToolError(
            code="tool_arguments_invalid_type",
            message="tool arguments must decode to a JSON object",
            category="validation",
            retryable=False,
        )
    return parsed


def validate_runtime_tool_arguments(tool: Any, arguments: dict[str, Any]) -> list[str]:
    schema = extract_runtime_tool_argument_schema(tool)
    if not isinstance(schema, dict):
        return []
    validator_kwargs: dict[str, Any] = {}
    contracts = getattr(tool, "_contracts", None)
    registry_for = getattr(contracts, "registry_for", None)
    if callable(registry_for):
        try:
            registry = registry_for(schema)
            if registry is not None:
                validator_kwargs["registry"] = registry
        except Exception:
            pass
    validator = Draft202012Validator(schema, **validator_kwargs)
    return [str(error.message) for error in validator.iter_errors(arguments)]


def extract_runtime_tool_argument_schema(tool: Any) -> dict[str, Any] | None:
    parameters = getattr(tool, "parameters", None)
    if isinstance(parameters, dict):
        return dict(parameters)

    tool_definition = tool_to_provider_definition(tool)
    if isinstance(tool_definition, dict):
        function_block = tool_definition.get("function")
        if isinstance(function_block, dict):
            schema = function_block.get("parameters")
            if isinstance(schema, dict):
                return dict(schema)
    return None


def tool_result_to_content(result: Any) -> str:
    if hasattr(result, "to_string") and callable(getattr(result, "to_string")):
        try:
            return str(result.to_string())
        except Exception:
            pass
    if isinstance(result, dict):
        return _serialize_payload(result)
    content = getattr(result, "content", None)
    error = getattr(result, "error", None)
    success = getattr(result, "success", None)
    payload: dict[str, Any] = {}
    if success is not None:
        payload["success"] = bool(success)
    if content is not None:
        payload["content"] = content
    if error:
        payload["error"] = str(error)
    if payload:
        return _serialize_payload(payload)
    return str(result)


def _serialize_payload(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


__all__ = [
    "StructuredToolRuntime",
    "StructuredToolLoopError",
    "RuntimeToolError",
    "complete_with_tools",
    "normalize_completion_tool_calls",
    "build_assistant_tool_call_message",
    "tool_to_provider_definition",
    "execute_runtime_tool",
    "parse_tool_arguments",
    "validate_runtime_tool_arguments",
    "extract_runtime_tool_argument_schema",
    "tool_result_to_content",
]
