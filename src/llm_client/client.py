import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Union, Literal, Type, Optional, Any, AsyncIterator

import numpy as np
import openai
from openai import AsyncOpenAI
from blake3 import blake3

from .cache import CacheSettings, build_cache_core
from .models import ModelProfile
from .rate_limit import Limiter
from .streaming import stream_as_sse, stream_to_sink, EventSink
from .types import LLMRequest, LLMResult, LLMEvent


class OpenAIClient:
    def __init__(
        self,
        model: Union[Type["ModelProfile"], str, None] = None,
        *,
        cache_dir: Union[str, Path, None] = None,
        responses_api_toggle: bool = False,
        cache_backend: Literal["qdrant", "pg_redis", "fs", None] = None,
        cache_collection: str | None = None,
        # optional DSNs for pg/redis/qdrant override if you want
        pg_dsn: str | None = None,
        redis_url: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        redis_ttl_seconds: int = 60 * 60 * 24,
        compress_pg: bool = True,
    ) -> None:
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.responses_api_toggle = responses_api_toggle
        self.default_cache_collection = cache_collection

        if self.cache_dir and cache_backend == "fs":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, type) and issubclass(model, ModelProfile):
            self.model = model
        elif isinstance(model, str):
            self.model = ModelProfile.get(model)
        else:
            raise ValueError("Model must be provided either as a ModelProfile subclass or a model key string.")

        self.limiter = Limiter(self.model)
        self.openai = AsyncOpenAI()

        backend_name = cache_backend or "none"
        self.cache = build_cache_core(
            CacheSettings(
                backend=backend_name,  # type: ignore[arg-type]
                client_type=self.model.category,
                default_collection=cache_collection,
                cache_dir=self.cache_dir,
                pg_dsn=pg_dsn,
                redis_url=redis_url,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                redis_ttl_seconds=redis_ttl_seconds,
                compress=compress_pg,
            )
        )

    async def warm_cache(self) -> None:
        await self.cache.warm()

    async def close(self) -> None:
        await self.cache.close()

    @staticmethod
    def encode_file(file_path: Union[str, Path]) -> dict:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        with open(file_path, "rb") as file:
            data = file.read()
        base64_encoded = base64.b64encode(data).decode("utf-8")

        if extension in {".jpg", ".jpeg", ".png", ".webp"}:
            return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{extension};base64,{base64_encoded}"
                    },
                }
        elif extension == ".pdf":
            return {
                    "type": "input_file",
                    "filename": file_path.name,
                    "file_data": base64_encoded,
                }
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    async def transcribe_pdf(self, file_path: Union[str, Path]) -> dict:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            raise ValueError("File does not exist or is not a PDF.")

        file = await self.openai.files.create(
            file=open(str(file_path), "rb"),
            purpose="user_data"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract text from this PDF document. "
                                "Do not include any metadata or file information, just the text content. "
                                "Your output should be the text extracted from the PDF."
                    },
                    {
                        "type": "input_file",
                        "file_id": file.id
                    },
                ]
            }
        ]

        request = LLMRequest(
            model=self.model.model_name,
            category="responses",
            input=messages,
            reasoning={"effort": "minimal"},
        )
        result = await self.invoke(request=request)
        return result.to_dict()

    async def transcribe_image(self, file_path: Union[str, Path]) -> dict:
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("File does not exist or is not an image.")

        with open(file_path, "rb") as file:
            data = file.read()
        base64_encoded = base64.b64encode(data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe the text from this image. Do not include any other explanations or metadata."
                                "Your output should be the image transcription only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_path.suffix[1:]};base64,{base64_encoded}"
                        },
                    },
                ],
            }
        ]

        response = await self.get_response(messages=messages, response_format="text")
        return response

    def check_reasoning_effort(self, params: dict, api_type: Literal["completions", "responses"]) -> dict:
        has_reasoning = "reasoning" in params
        has_reasoning_effort = "reasoning_effort" in params

        if not (has_reasoning or has_reasoning_effort):
            return params

        if not self.model.reasoning_model:
            raise ValueError("Model does not support reasoning, but reasoning parameters were provided.")

        effort = None

        if has_reasoning:
            reasoning_val = params["reasoning"]
            if not isinstance(reasoning_val, dict):
                raise ValueError("`reasoning` must be an object like {'effort': '<level>'}.")
            effort = reasoning_val.get("effort")

        if has_reasoning_effort:
            reff = params.get("reasoning_effort")
            if effort is not None and reff is not None and reff != effort:
                raise ValueError("Provide only one of `reasoning` or `reasoning_effort`, or ensure they match.")
            effort = reff if effort is None else effort

        if effort not in self.model.reasoning_efforts:
            raise ValueError(f"Invalid reasoning effort. Choose from: {self.model.reasoning_efforts}")

        if api_type == "responses":
            params.pop("reasoning_effort", None)
            params["reasoning"] = {"effort": effort}
        elif api_type == "completions":
            params.pop("reasoning", None)
            params["reasoning_effort"] = effort

        return params

    def _build_request(self, **kwargs) -> LLMRequest:
        messages = kwargs.pop("messages", None)
        input_value = kwargs.pop("input", None)
        response_format = kwargs.pop("response_format", None)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        reasoning = kwargs.pop("reasoning", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        stream = bool(kwargs.pop("stream", False))
        stream_options = kwargs.pop("stream_options", None)
        kwargs.pop("stream_mode", None)

        category: Literal["completions", "embeddings", "responses"]
        if self.responses_api_toggle:
            category = "responses"
        else:
            category = self.model.category  # type: ignore[assignment]

        return LLMRequest(
            model=self.model.model_name,
            category=category,
            messages=messages,
            input=input_value,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            stream=stream,
            stream_options=stream_options,
            extra=kwargs,
        )

    @staticmethod
    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _normalize_messages(self, messages: Any, input_value: Any) -> list:
        if messages is None:
            messages = input_value
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return messages
        raise ValueError("Messages must be a list of dicts or a string.")

    def _normalize_response_format_for_cache(self, response_format: Any) -> Any:
        if response_format is None:
            return None
        if isinstance(response_format, dict):
            return response_format
        if isinstance(response_format, str):
            return response_format
        return {
            "type": "json_schema",
            "schema_name": getattr(response_format, "__name__", str(response_format)),
        }

    def _cache_payload(self, request: LLMRequest) -> dict:
        return {
            "model": request.model,
            "category": request.category,
            "messages": request.messages,
            "input": request.input,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "response_format": self._normalize_response_format_for_cache(request.response_format),
            "reasoning": request.reasoning,
            "reasoning_effort": request.reasoning_effort,
            "params": request.extra,
        }

    def _default_identifier(self, request: LLMRequest, hash_as_identifier: bool) -> str:
        if not hash_as_identifier:
            return blake3(str(uuid.uuid4()).encode("utf-8")).hexdigest()
        payload = self._cache_payload(request)
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return blake3(payload_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _tool_call_to_dict(tool_call: Any) -> dict:
        if isinstance(tool_call, dict):
            return tool_call
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump()
        if hasattr(tool_call, "dict"):
            return tool_call.dict()
        if hasattr(tool_call, "to_dict"):
            return tool_call.to_dict()
        function = getattr(tool_call, "function", None)
        function_dict = None
        if function is not None:
            function_dict = {
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None),
            }
        return {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None),
            "function": function_dict,
        }

    @staticmethod
    def _message_to_dict(message: Any) -> dict | None:
        if message is None:
            return None
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if hasattr(message, "dict"):
            return message.dict()
        if hasattr(message, "to_dict"):
            return message.to_dict()
        return {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }

    def _usage_obj_to_dict(self, usage: Any) -> dict:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "to_dict"):
            return usage.to_dict()
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "__dict__"):
            return dict(usage.__dict__)
        return {}

    def _normalize_usage(self, usage: dict) -> dict:
        usage = dict(usage or {})
        for key in ("completion_tokens_details", "prompt_tokens_details"):
            details = usage.get(key)
            if details is None or isinstance(details, dict):
                continue
            if hasattr(details, "dict"):
                usage[key] = details.dict()
            elif hasattr(details, "model_dump"):
                usage[key] = details.model_dump()
            else:
                usage[key] = dict(details)
        return self.model.parse_usage(usage)

    @staticmethod
    def _parsed_output(parsed: Any) -> Any:
        if parsed is None:
            return None
        if isinstance(parsed, dict):
            return parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if hasattr(parsed, "dict"):
            return parsed.dict()
        return parsed

    def _extract_tool_calls(self, message: Any) -> list[dict] | None:
        if message is None:
            return None
        tool_calls = self._get_attr(message, "tool_calls")
        if not tool_calls:
            return None
        return [self._tool_call_to_dict(tc) for tc in tool_calls]

    def _prepare_chat_params(self, request: LLMRequest) -> tuple[dict, bool]:
        params = dict(request.extra)
        params["model"] = request.model
        params["messages"] = self._normalize_messages(request.messages, request.input)

        if request.tools is not None:
            if not self.model.function_calling_support:
                raise ValueError("Model does not support tool calling, but tools were provided.")
            params["tools"] = request.tools
        if request.tool_choice is not None:
            params["tool_choice"] = request.tool_choice
        if request.reasoning is not None:
            params["reasoning"] = request.reasoning
        if request.reasoning_effort is not None:
            params["reasoning_effort"] = request.reasoning_effort

        params = self.check_reasoning_effort(params, "completions")

        response_format = request.response_format
        use_parse = False

        if response_format is None:
            params["response_format"] = {"type": "text"}
        elif isinstance(response_format, str):
            if response_format == "text":
                params["response_format"] = {"type": "text"}
            elif response_format == "json_object":
                if "json" not in str(params["messages"]).lower():
                    raise ValueError("Context doesn't contain 'json' keyword which is required for JSON mode.")
                params["response_format"] = {"type": "json_object"}
            else:
                params["response_format"] = response_format
        elif isinstance(response_format, dict):
            params["response_format"] = response_format
        else:
            use_parse = True
            params["response_format"] = response_format

        return params, use_parse

    def _prepare_responses_params(self, request: LLMRequest) -> dict:
        params = dict(request.extra)
        params["model"] = request.model
        if request.input is None:
            raise ValueError("'input' is required for responses.")
        params["input"] = request.input
        if request.reasoning is not None:
            params["reasoning"] = request.reasoning
        if request.reasoning_effort is not None:
            params["reasoning_effort"] = request.reasoning_effort

        params = self.check_reasoning_effort(params, "responses")

        if request.response_format is not None:
            raise ValueError("response_format is not supported with 'responses' endpoint YET.")

        return params

    def _prepare_embeddings_params(self, request: LLMRequest) -> dict:
        params = dict(request.extra)
        params["model"] = request.model
        if request.input is None:
            raise ValueError("Input is required for embeddings.")
        params["input"] = request.input
        if not params.get("encoding_format"):
            params["encoding_format"] = "base64"
        return params

    async def _invoke_responses(self, request: LLMRequest) -> tuple[LLMResult, Any]:
        params = self._prepare_responses_params(request)

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"
        response = None

        async with self.limiter.limit(tokens=self.model.count_tokens(params["input"]), requests=1) as limit_context:
            try:
                response = await self.openai.responses.create(**params)
                output = response.output_text
                usage_raw = self._usage_obj_to_dict(response.usage)
                status, error = 200, "OK"
                usage = self._normalize_usage(usage_raw)
                limit_context.output_tokens = usage.get("output_tokens", 0)
            except openai.APIConnectionError as exc:
                status, error = 500, exc.__cause__
                print(f"API Connection Error in Responses: {error}")
            except openai.RateLimitError as exc:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Responses: {error} - {exc}")
            except openai.APIStatusError as exc:
                status, error = exc.status_code, exc.response
                print(f"API Status Error in Responses: {error}")
            finally:
                usage = usage or {}

        result = LLMResult(
            output=output,
            usage=usage,
            status=status,
            error=error,
            params=params,
        )
        return result, response

    async def _invoke_completions(self, request: LLMRequest) -> tuple[LLMResult, Any]:
        params, use_parse = self._prepare_chat_params(request)

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"
        response = None
        message = None
        tool_calls = None

        input_tokens = self.model.count_tokens(params["messages"])

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_context:
            try:
                if use_parse:
                    response = await self.openai.beta.chat.completions.parse(**params)
                    message = response.choices[0].message
                    parsed = self._get_attr(message, "parsed")
                    output = self._parsed_output(parsed)
                else:
                    response = await self.openai.chat.completions.create(**params)
                    message = response.choices[0].message
                    output = self._get_attr(message, "content")
                    rf = params.get("response_format")
                    if isinstance(rf, dict) and rf.get("type") in ["json_object", "json_schema"] and output:
                        output = json.loads(output)
                tool_calls = self._extract_tool_calls(message)
                message_dict = self._message_to_dict(message)
                usage = self._normalize_usage(self._usage_obj_to_dict(response.usage))
                status, error = 200, "OK"
                limit_context.output_tokens = usage.get("output_tokens", 0)
            except openai.APIConnectionError as exc:
                status, error = 500, exc.__cause__
                print(f"API Connection Error in Completions: {error}")
                message_dict = None
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Completions: {error}")
                message_dict = None
            except openai.APIStatusError as exc:
                status, error = exc.status_code, exc.response
                print(f"API Status Error in Completions: {error}")
                message_dict = None
            finally:
                usage = usage or {}

        result = LLMResult(
            output=output,
            usage=usage,
            status=status,
            error=error,
            params=params,
            tool_calls=tool_calls,
            message=message_dict,
        )
        return result, response

    async def _invoke_embeddings(self, request: LLMRequest) -> tuple[LLMResult, Any]:
        params = self._prepare_embeddings_params(request)

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"
        response = None

        input_tokens = self.model.count_tokens(params["input"])

        async with self.limiter.limit(tokens=input_tokens, requests=1):
            try:
                response = await self.openai.embeddings.create(**params)
                output = [d.embedding for d in response.data]
                if params["encoding_format"] == "base64":
                    output = [
                        np.frombuffer(base64.b64decode(embedding), dtype=np.float32).tolist()
                        for embedding in output
                        if isinstance(embedding, str)
                    ]
                output = output[0] if len(output) == 1 else output
                usage = self._normalize_usage(self._usage_obj_to_dict(response.usage))
                status, error = 200, "OK"
            except openai.APIConnectionError as exc:
                status, error = 500, exc.__cause__
                print(f"API Connection Error in Embeddings: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Embeddings: {error}")
            except openai.APIStatusError as exc:
                status, error = exc.status_code, exc.response
                print(f"API Status Error in Embeddings: {error}")
            finally:
                usage = usage or {}

        result = LLMResult(
            output=output,
            usage=usage,
            status=status,
            error=error,
            params=params,
        )
        return result, response

    async def _invoke_once(self, request: LLMRequest) -> tuple[LLMResult, Any]:
        if request.category == "responses":
            return await self._invoke_responses(request)
        if request.category == "completions":
            return await self._invoke_completions(request)
        if request.category == "embeddings":
            return await self._invoke_embeddings(request)
        raise ValueError(f"Unknown request category: {request.category}")

    def _llm_result_from_cached(self, cached: dict) -> LLMResult:
        return LLMResult(
            output=cached.get("output"),
            usage=cached.get("usage", {}),
            status=cached.get("status", 0),
            error=cached.get("error", ""),
            params=cached.get("params", {}),
            tool_calls=cached.get("tool_calls"),
            message=cached.get("message"),
        )

    def _result_to_cache_dict(self, result: LLMResult, identifier: str, body: dict | None) -> dict:
        params_for_cache = dict(result.params)
        params_for_cache.pop("stream", None)
        params_for_cache.pop("stream_options", None)
        rf = params_for_cache.get("response_format")
        if rf and not isinstance(rf, dict):
            params_for_cache["response_format"] = {
                "type": "json_schema",
                "schema_name": getattr(rf, "__name__", str(rf)),
            }

        response = {
            "params": params_for_cache,
            "output": result.output,
            "usage": result.usage,
            "status": result.status,
            "error": result.error,
            "identifier": identifier,
            "body": body,
        }
        if result.tool_calls is not None:
            response["tool_calls"] = result.tool_calls
        if result.message is not None:
            response["message"] = result.message
        return response

    def _accumulate_tool_calls(self, tool_calls: list[dict], deltas: Any) -> list[dict]:
        updates: list[dict] = []
        for delta in deltas:
            index = self._get_attr(delta, "index")
            if index is None:
                index = len(tool_calls)
            while len(tool_calls) <= index:
                tool_calls.append({
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                })
            call = tool_calls[index]
            call_id = self._get_attr(delta, "id")
            call_type = self._get_attr(delta, "type")
            function = self._get_attr(delta, "function")
            name = self._get_attr(function, "name") if function is not None else None
            arguments = self._get_attr(function, "arguments") if function is not None else None

            if call_id:
                call["id"] = call_id
            if call_type:
                call["type"] = call_type
            if name:
                call.setdefault("function", {})["name"] = name
            if arguments:
                call.setdefault("function", {}).setdefault("arguments", "")
                call["function"]["arguments"] += arguments

            updates.append({
                "index": index,
                "id": call_id,
                "type": call_type,
                "name": name,
                "arguments": arguments,
            })
        return updates

    @staticmethod
    def _finalize_tool_calls(tool_calls: list[dict]) -> list[dict] | None:
        finalized = []
        for call in tool_calls:
            function = call.get("function") or {}
            if not call.get("id") and not function.get("name"):
                continue
            if function.get("arguments") is None:
                function["arguments"] = ""
            call["function"] = function
            finalized.append(call)
        return finalized or None

    async def _events_from_cached(self, result: LLMResult) -> AsyncIterator[LLMEvent]:
        yield LLMEvent("meta", {"model": self.model.model_name, "cache_hit": True})
        if isinstance(result.output, str) and result.output:
            yield LLMEvent("text_delta", {"text": result.output})
        if result.usage:
            yield LLMEvent("usage", {"usage": result.usage})
        yield LLMEvent("done", {"result": result.to_dict()})

    async def _stream_completions(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        params, use_parse = self._prepare_chat_params(request)
        if use_parse:
            raise ValueError("Streaming with structured response_format is not supported.")
        params["stream"] = True
        if request.stream_options is not None:
            params["stream_options"] = request.stream_options
        elif "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        input_tokens = self.model.count_tokens(params["messages"])
        output = ""
        tool_calls: list[dict] = []

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_context:
            listener = await self.openai.chat.completions.create(**params)
            yield LLMEvent("meta", {"model": params["model"], "category": "completions"})
            async for chunk in listener:
                if not chunk.choices and chunk.usage:
                    usage_raw = self._usage_obj_to_dict(chunk.usage)
                    usage = self._normalize_usage(usage_raw)
                    limit_context.output_tokens = usage.get("output_tokens", 0)
                    yield LLMEvent("usage", {"usage": usage})
                    final_tool_calls = self._finalize_tool_calls(tool_calls)
                    result = LLMResult(
                        output=output,
                        usage=usage,
                        status=200,
                        error="OK",
                        params=params,
                        tool_calls=final_tool_calls,
                    )
                    yield LLMEvent("done", {"result": result.to_dict()})
                    return

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                token = self._get_attr(delta, "content")
                if token:
                    output += token
                    yield LLMEvent("text_delta", {"text": token})

                tool_deltas = self._get_attr(delta, "tool_calls")
                if tool_deltas:
                    for update in self._accumulate_tool_calls(tool_calls, tool_deltas):
                        yield LLMEvent("tool_call_delta", update)

        final_tool_calls = self._finalize_tool_calls(tool_calls)
        result = LLMResult(
            output=output,
            usage={},
            status=200,
            error="OK",
            params=params,
            tool_calls=final_tool_calls,
        )
        yield LLMEvent("done", {"result": result.to_dict()})

    def _stream_once(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        if request.category != "completions":
            raise ValueError("Streaming is only supported for completions.")
        return self._stream_completions(request)

    async def invoke(
        self,
        request: LLMRequest | None = None,
        *,
        identifier: str | None = None,
        attempts: int = 3,
        backoff: int = 1,
        body: dict | None = None,
        cache_response: bool = False,
        return_response: bool = False,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        log_errors: bool = True,
        timeout: float | None = None,
        hash_as_identifier: bool = True,
        cache_collection: str | None = None,
        **kwargs,
    ) -> LLMResult:
        req = request or self._build_request(**kwargs)
        if req.stream:
            raise ValueError("Streaming requests should use stream().")

        effective_collection = cache_collection or self.default_cache_collection
        if not identifier:
            identifier = self._default_identifier(req, hash_as_identifier)

        if cache_response:
            cached, _ = await self.cache.get_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                only_ok=True,
                collection=effective_collection,
            )
            if cached:
                result = self._llm_result_from_cached(cached)
                result.metadata["identifier"] = cached.get("identifier", identifier)
                result.metadata["body"] = cached.get("body")
                result.metadata["cache_hit"] = True
                return result

        attempts_left = attempts
        current_backoff = backoff
        response = None
        result: LLMResult | None = None

        while attempts_left > 0:
            try:
                if timeout is None:
                    result, response = await self._invoke_once(req)
                else:
                    result, response = await asyncio.wait_for(self._invoke_once(req), timeout=timeout)
            except asyncio.TimeoutError:
                attempts_left -= 1
                if attempts_left == 0:
                    timeout_result = LLMResult(
                        output=None,
                        usage={},
                        status=408,
                        error="Timeout",
                        params={},
                    )
                    timeout_result.metadata["identifier"] = identifier
                    timeout_result.metadata["body"] = body
                    return timeout_result
                await asyncio.sleep(current_backoff)
                current_backoff *= 2
                continue

            if result.status < 500:
                break

            attempts_left -= 1
            if attempts_left == 0:
                break

            await asyncio.sleep(current_backoff)
            current_backoff *= 2

        assert result is not None

        body_with_completion = None
        if body is not None:
            body_with_completion = dict(body)
            body_with_completion["completion"] = result.output

        result.metadata["identifier"] = identifier
        result.metadata["body"] = body_with_completion
        if return_response:
            result.metadata["response"] = response

        if cache_response:
            cache_record = self._result_to_cache_dict(result, identifier, body_with_completion)
            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=cache_record,
                model_name=self.model.model_name,
                log_errors=log_errors,
                collection=effective_collection,
            )

        return result

    def stream(
        self,
        request: LLMRequest | None = None,
        *,
        identifier: str | None = None,
        attempts: int = 3,
        backoff: int = 1,
        body: dict | None = None,
        cache_response: bool = False,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        log_errors: bool = True,
        hash_as_identifier: bool = True,
        cache_collection: str | None = None,
        **kwargs,
    ) -> AsyncIterator[LLMEvent]:
        req = request or self._build_request(**kwargs)
        req.stream = True
        effective_collection = cache_collection or self.default_cache_collection
        if not identifier:
            identifier = self._default_identifier(req, hash_as_identifier)

        async def generator() -> AsyncIterator[LLMEvent]:
            if cache_response:
                cached, _ = await self.cache.get_cached(
                    identifier,
                    rewrite_cache=rewrite_cache,
                    regen_cache=regen_cache,
                    only_ok=True,
                    collection=effective_collection,
                )
                if cached:
                    result = self._llm_result_from_cached(cached)
                    result.metadata["identifier"] = cached.get("identifier", identifier)
                    result.metadata["body"] = cached.get("body")
                    result.metadata["cache_hit"] = True
                    async for event in self._events_from_cached(result):
                        yield event
                    return

            attempts_left = attempts
            current_backoff = backoff

            while attempts_left > 0:
                emitted_any = False
                try:
                    async for event in self._stream_once(req):
                        emitted_any = True
                        if event.type == "done":
                            result_dict = event.data.get("result", {})
                            result = self._llm_result_from_cached(result_dict)
                            body_with_completion = None
                            if body is not None:
                                body_with_completion = dict(body)
                                body_with_completion["completion"] = result.output
                            result.metadata["identifier"] = identifier
                            result.metadata["body"] = body_with_completion
                            done_event = LLMEvent("done", {"result": result.to_dict()})
                            yield done_event

                            if cache_response:
                                cache_record = self._result_to_cache_dict(
                                    result,
                                    identifier,
                                    body_with_completion,
                                )
                                await self.cache.put_cached(
                                    identifier,
                                    rewrite_cache=rewrite_cache,
                                    regen_cache=regen_cache,
                                    response=cache_record,
                                    model_name=self.model.model_name,
                                    log_errors=log_errors,
                                    collection=effective_collection,
                                )
                            return

                        yield event

                    return
                except Exception as exc:
                    if emitted_any:
                        error_result = LLMResult(
                            output=None,
                            usage={},
                            status=500,
                            error=str(exc),
                            params={},
                        )
                        error_result.metadata["identifier"] = identifier
                        error_result.metadata["body"] = body
                        yield LLMEvent("error", {"error": str(exc), "status": 500})
                        yield LLMEvent("done", {"result": error_result.to_dict()})
                        return

                    attempts_left -= 1
                    if attempts_left == 0:
                        error_result = LLMResult(
                            output=None,
                            usage={},
                            status=500,
                            error=str(exc),
                            params={},
                        )
                        error_result.metadata["identifier"] = identifier
                        error_result.metadata["body"] = body
                        yield LLMEvent("error", {"error": str(exc), "status": 500})
                        yield LLMEvent("done", {"result": error_result.to_dict()})
                        return

                    await asyncio.sleep(current_backoff)
                    current_backoff *= 2

        return generator()

    def stream_sse(self, request: LLMRequest | None = None, **kwargs) -> AsyncIterator[str]:
        return stream_as_sse(self.stream(request=request, **kwargs))

    async def stream_to_sink(
        self,
        sink: EventSink,
        request: LLMRequest | None = None,
        **kwargs,
    ) -> None:
        await stream_to_sink(self.stream(request=request, **kwargs), sink)

    async def get_response(
        self,
        identifier: str | None = None,
        attempts: int = 3,
        backoff: int = 1,
        body: dict | None = None,
        cache_response: bool = False,
        return_response: bool = False,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        log_errors: bool = True,
        timeout: float | None = None,
        hash_as_identifier: bool = True,
        cache_collection: str | None = None,
        **kwargs,
    ) -> dict:
        if kwargs.get("stream"):
            raise ValueError("Streaming is now handled by stream() or stream_sse().")

        result = await self.invoke(
            identifier=identifier,
            attempts=attempts,
            backoff=backoff,
            body=body,
            cache_response=cache_response,
            return_response=return_response,
            rewrite_cache=rewrite_cache,
            regen_cache=regen_cache,
            log_errors=log_errors,
            timeout=timeout,
            hash_as_identifier=hash_as_identifier,
            cache_collection=cache_collection,
            **kwargs,
        )
        return result.to_dict()

    async def upload_batch_input_file(self, file_path: Union[str, Path]) -> dict:
        """
        Uploads a JSONL file for batch processing.
        """
        file_path = Path(file_path)
        if not file_path.exists():
             raise ValueError(f"File {file_path} does not exist.")

        file_obj = await self.openai.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        return file_obj.dict()

    async def create_batch_job(
        self,
        input_file_id: str,
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"] = "/v1/chat/completions",
        completion_window: Literal["24h"] = "24h",
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Create a batch job using the OpenAI Batch API.
        """
        batch = await self.openai.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata
        )
        return batch.dict()

    async def retrieve_batch_job(self, batch_id: str) -> dict:
        """
        Retrieve details of a batch job.
        """
        batch = await self.openai.batches.retrieve(batch_id)
        return batch.dict()

    async def cancel_batch_job(self, batch_id: str) -> dict:
        """
        Cancel a batch job.
        """
        batch = await self.openai.batches.cancel(batch_id)
        return batch.dict()

    async def list_batch_jobs(self, limit: int = 20, after: str = None) -> dict:
        """
        List your batch jobs.
        """
        batches = await self.openai.batches.list(limit=limit, after=after)
        return batches.dict()

    async def download_batch_results(self, file_id: str) -> bytes:
        """
        Download the result file content.
        """
        content = await self.openai.files.content(file_id)
        return content.read()
