import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Union, Literal, Type, Optional, Any, AsyncGenerator

import numpy as np
import openai
from openai import AsyncOpenAI
from blake3 import blake3

from .cache import CacheSettings, build_cache_core
from .models import ModelProfile
from .rate_limit import Limiter
from .streaming import PusherStreamer, format_sse_event
from .streams import StreamResponse
from .types import RequestParams

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

        if self.responses_api_toggle:
            self._call_model = self._call_responses
        elif self.model.category == "completions":
            self._call_model = self._call_completions
        elif self.model.category == "embeddings":
            self._call_model = self._call_embeddings

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

        result, _ = await self._call_responses(input=messages, reasoning={"effort": "minimal"})
        return result

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

    async def _call_responses(self, **params) -> Union[dict, any]:
        assert params.get("input"), "'input' is required for completions."
        
        params["model"] = self.model.model_name
        params = self.check_reasoning_effort(params, "responses")

        if "response_format" in params:
            raise ValueError("response_format is not supported with 'responses' endpoint YET.")

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

        async with self.limiter.limit(tokens=self.model.count_tokens(params["input"]),
                                      requests=1) as limit_context:
            try:
                response = await self.openai.responses.create(**params)
                output = response.output_text
                usage_raw = response.usage.to_dict()
                status, error = 200, "OK"
                usage = self.model.parse_usage(usage_raw)
                limit_context.output_tokens = usage.get("output_tokens", 0)
            except openai.APIConnectionError as e:
                status, error = 500, e.__cause__
                print(f"API Connection Error in Completions: {error}")
            except openai.RateLimitError as e:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Completions: {error} - {e}")
            except openai.APIStatusError as e:
                status, error = e.status_code, e.response
                print(f"API Status Error in Completions: {error}")
            finally:
                usage = usage or {}
        result = dict(params=params, output=output, usage=usage, status=status, error=error)
        return result, response

    async def _call_completions(self, **params) -> Union[dict, any]:
        assert params.get("messages"), "Messages are required for completions."
        params["model"] = self.model.model_name

        params = self.check_reasoning_effort(params, "completions")

        if params.get("response_format"):
            if params["response_format"] == "text":
                params["response_format"] = {"type": "text"}
            elif params["response_format"] == "json_object":
                if "json" not in str(params["messages"]).lower():
                    raise ValueError("Context doesn't contain 'json' keyword which is required for JSON mode.")
                params["response_format"] = {"type": "json_object"}
        else:
            params["response_format"] = {"type": "text"}

        if isinstance(params["messages"], str):
            params["messages"] = [{"role": "user", "content": params["messages"]}]

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

        # Check for stream mode
        is_stream = params.get("stream", False)
        stream_mode = params.pop("stream_mode", "pusher") if is_stream else None
        
        # If streaming, we return the generator/StreamResponse differently 
        # inside the try/except block to ensure we can handle errors during stream initiation.

        async with self.limiter.limit(
            tokens=self.model.count_tokens(params["messages"]), requests=1
        ) as limit_context:
            try:
                if is_stream:
                    params["stream_options"] = {"include_usage": True}
                    
                    if stream_mode == "sse":
                        # Return StreamResponse object which wraps the stream
                         async def sse_generator():
                            nonlocal output, usage, status, error
                            try:
                                yield format_sse_event("meta", json.dumps({
                                    "model": params["model"],
                                    "stream_mode": "sse"
                                }))
                                
                                listener = await self.openai.chat.completions.create(**params)
                                # Wrap with StreamResponse if useful or manually handle
                                # For backward compat with existing 'sse' logic in get_response, we do manual here
                                # But ideally we should use StreamResponse.
                                # Let's keep existing logic for now but wrapped cleanly.
                                
                                async for chunk in listener:
                                    if not chunk.choices and chunk.usage:
                                        usage = dict(chunk.usage)
                                        usage["completion_tokens_details"] = usage["completion_tokens_details"].dict()
                                        usage["prompt_tokens_details"] = usage["prompt_tokens_details"].dict()
                                        usage = self.model.parse_usage(usage)
                                        status, error = 200, "OK"
                                        yield format_sse_event("done", json.dumps({
                                            "usage": usage,
                                            "status": status,
                                            "output": output
                                        }))
                                        break
                                    
                                    delta = chunk.choices[0].delta.content
                                    if delta is not None:
                                        output = (output or "") + delta
                                        yield format_sse_event("token", delta)
                                        
                            except Exception as e:
                                error = str(e)
                                status = 500
                                yield format_sse_event("error", json.dumps({"error": error, "status": status}))

                         return sse_generator(), None

                    elif stream_mode == "raw": 
                        # New Raw Stream Mode for Agents using StreamResponse
                        # We don't wait for completion here, we return the StreamResponse immediately
                        response_stream = await self.openai.chat.completions.create(**params)
                        # We return the stream object itself as the "result" payload temporarily
                        # The caller (get_response) knows how to handle this.
                        # However, to fit the (dict, response) signature, we return a special dict.
                        return {"stream": response_stream, "status": 200, "stream_mode": "raw"}, None

                    else:
                        # Pusher streaming mode (default legacy)
                        output = ""
                        async with PusherStreamer(channel=params.pop("channel", str(uuid.uuid4()))) as streamer:
                            listener = await self.openai.chat.completions.create(**params)
                            await streamer.push_event("new-response", "")
                            async for chunk in listener:
                                if not chunk.choices and chunk.usage:
                                    usage = dict(chunk.usage)
                                    usage["completion_tokens_details"] = usage["completion_tokens_details"].dict()
                                    usage["prompt_tokens_details"] = usage["prompt_tokens_details"].dict()
                                    status, error = 200, "OK"
                                    break
                                token = chunk.choices[0].delta.content
                                if token is not None:
                                    await streamer.push_event("new-token", token)
                                    output = (output or "") + token
                            await streamer.push_event("response-finished", error)

                else:
                    # Non-streaming
                    if isinstance(params["response_format"], dict):
                        response = await self.openai.chat.completions.create(**params)
                        output = response.choices[0].message.content
                        tool_calls = response.choices[0].message.tool_calls
                        if params["response_format"]["type"] in ["json_object", "json_schema"]:
                            output = json.loads(output)
                    else:
                        response = await self.openai.beta.chat.completions.parse(**params)
                        output = response.choices[0].message.parsed.dict()
                        tool_calls = response.choices[0].message.tool_calls

                    usage = response.usage.to_dict()
                    status, error = 200, "OK"

                usage = self.model.parse_usage(usage)
                limit_context.output_tokens = usage.get("output_tokens", 0)
            except openai.APIConnectionError as exc:
                status, error = 500, exc.__cause__
                print(f"API Connection Error in Completions: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Completions: {error}")
            except openai.APIStatusError as exc:
                status, error = exc.status_code, exc.response
                print(f"API Status Error in Completions: {error}")
            finally:
                usage = usage or {}
        
        result = dict(params=params, output=output, usage=usage, status=status, error=error)
        if tool_calls:
            result["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc.dict() 
                for tc in tool_calls
            ]
        return result, response

    async def _call_embeddings(self, **params) -> Union[dict, any]:
        assert params.get("input"), "Input is required for embeddings."
        params["model"] = self.model.model_name
        if not params.get("encoding_format"):
            params["encoding_format"] = "base64"

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

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
                usage = self.model.parse_usage(response.usage.to_dict())
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
        result = dict(params=params, output=output, usage=usage, status=status, error=error)
        return result, response

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
    ) -> Union[dict, AsyncGenerator, StreamResponse]:
        """
        Main entry point for getting responses from the LLM.
        Supports:
        - Caching (checking and updating)
        - Retries with backoff
        - Streaming (Pusher, SSE, and new Raw/StreamResponse)
        """
        # Resolve effective cache collection
        effective_collection = cache_collection or self.default_cache_collection

        # Calculate identifier (cache key)
        if not identifier:
            if hash_as_identifier:
                content = kwargs.get("input", kwargs.get("messages", ""))
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, sort_keys=True)
                identifier = blake3(str(content).encode("utf-8")).hexdigest()
            else:
                identifier = blake3(str(uuid.uuid4()).encode("utf-8")).hexdigest()

        # Check Cache
        if cache_response:
            cached, _ = await self.cache.get_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                only_ok=True,
                collection=effective_collection,
            )
            if cached:
                return cached

        result: dict = {}
        attempts_left = attempts
        current_backoff = backoff
        response = None

        while attempts_left > 0:
            try:
                # Perform the call
                if timeout is None:
                    result, response = await self._call_model(**kwargs)
                else:
                    result, response = await asyncio.wait_for(self._call_model(**kwargs), timeout=timeout)
                
                # Special handling for Streaming Modes
                if kwargs.get("stream"):
                    # SSE Mode: result is an AsyncGenerator
                    if kwargs.get("stream_mode") == "sse":
                        return result
                    
                    # Raw Mode: result is {"stream": AsyncStream, ...}
                    # We wrap it in StreamResponse and return immediately
                    if isinstance(result, dict) and result.get("stream_mode") == "raw":
                         return StreamResponse(result["stream"], stream_mode="raw")

            except asyncio.TimeoutError:
                attempts_left -= 1
                if attempts_left == 0:
                    return {"status": 408, "error": "Timeout", "output": None, "usage": {}}
                await asyncio.sleep(current_backoff)
                current_backoff *= 2
                continue
            
            # Check success
            if result.get("status", 500) < 500:
                break

            attempts_left -= 1
            if attempts_left == 0:
                return result

            await asyncio.sleep(current_backoff)
            current_backoff *= 2

        # Post-processing for non-streaming successes
        result.update({"identifier": identifier, "body": body})
        if result.get("body"):
            result["body"]["completion"] = result.get("output")

        # Save to Cache
        if cache_response and result.get("status") == 200:
            rf = result.get("params", {}).get("response_format", {})
            if rf and not isinstance(rf, dict):
                result["params"]["response_format"] = {
                    "type": "json_schema",
                    "schema_name": getattr(rf, "__name__", str(rf)),
                }

            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=result,
                model_name=self.model.model_name,
                log_errors=log_errors,
                collection=effective_collection,
            )

        if return_response:
            result["response"] = response

        return result

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


__all__ = ["OpenAIClient"]
