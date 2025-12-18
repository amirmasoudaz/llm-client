import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Union, Literal, Type

import numpy as np
import openai
from openai import AsyncOpenAI
from blake3 import blake3

from .cache import CacheSettings, build_cache_core
from .models import ModelProfile
from .rate_limit import Limiter
from .streaming import PusherStreamer


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

        if self.cache_dir and cache_backend == "fs":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, type) and issubclass(model, ModelProfile):
            self.model = model
        elif isinstance(model, str):
            self.model = ModelProfile.get(model)
        else:
            raise ValueError("Model must be provided either as a ModelProfile subclass or a model key string.")

        if self.model.category == "completions":
            self._call_model = self._call_completions
        elif self.model.category == "embeddings":
            self._call_model = self._call_embeddings

        self.limiter = Limiter(self.model)
        self.openai = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(256)

        backend_name = cache_backend or "none"
        self.cache = build_cache_core(
            CacheSettings(
                backend=backend_name,  # type: ignore[arg-type]
                client_type=self.model.category,
                collection=cache_collection,
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

    async def _call_completions(self, **params) -> dict:
        assert params.get("messages"), "Messages are required for completions."
        params["model"] = self.model.model_name

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

        async with self.limiter.limit(
            tokens=self.model.count_tokens(params["messages"]), requests=1
        ) as limit_context:
            try:
                if params.get("stream", False):
                    output = ""
                    params["stream_options"] = {"include_usage": True}
                    async with PusherStreamer(channel=params.pop("channel", str(uuid.uuid4()))) as streamer:
                        listener = await self.openai.chat.completions.create(**params)
                        await streamer.push_event("new-response", "")
                        async for chunk in listener:
                            if not chunk.choices and chunk.usage:
                                usage = dict(chunk.usage)
                                usage["completion_tokens_details"] = usage["completion_tokens_details"].dict()
                                usage["prompt_tokens_details"] = usage["prompt_tokens_details"].dict()
                                error, status = "OK", 200
                                break
                            token = chunk.choices[0].delta.content
                            if token is not None:
                                await streamer.push_event("new-token", token)
                                output += token
                        await streamer.push_event("response-finished", error)
                else:
                    if isinstance(params["response_format"], dict):
                        response = await self.openai.chat.completions.create(**params)
                        output = response.choices[0].message.content
                        if params["response_format"]["type"] == "json_object":
                            output = json.loads(output)
                    else:
                        response = await self.openai.beta.chat.completions.parse(**params)
                        output = response.choices[0].message.parsed.dict()
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
        return dict(params=params, output=output, usage=usage, status=status, error=error)

    async def _call_embeddings(self, **params) -> dict:
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
        return dict(params=params, output=output, usage=usage, status=status, error=error)

    async def get_response(
        self,
        identifier: str | None = None,
        attempts: int = 3,
        backoff: int = 1,
        body: dict | None = None,
        cache_response: bool = False,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        log_errors: bool = True,
        timeout: float | None = None,
        hash_as_identifier: bool = True,
        **kwargs,
    ) -> dict:
        if not identifier:
            if hash_as_identifier:
                content = kwargs.get("input", kwargs.get("messages", ""))
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, sort_keys=True)
                identifier = blake3(str(content).encode("utf-8")).hexdigest()
            else:
                identifier = blake3(str(uuid.uuid4()).encode("utf-8")).hexdigest()

        if cache_response:
            cached, _ = await self.cache.get_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                only_ok=True,
            )
            if cached:
                return cached

        response: dict = {}
        attempts_left = attempts
        current_backoff = backoff

        while attempts_left > 0:
            try:
                if timeout is None:
                    response = await self._call_model(**kwargs)
                else:
                    response = await asyncio.wait_for(self._call_model(**kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                attempts_left -= 1
                if attempts_left == 0:
                    return {"status": 408, "error": "Timeout", "output": None, "usage": {}}
                await asyncio.sleep(current_backoff)
                current_backoff *= 2
                continue

            if response["status"] < 500:
                break

            attempts_left -= 1
            if attempts_left == 0:
                return response

            await asyncio.sleep(current_backoff)
            current_backoff *= 2

        response.update({"identifier": identifier, "body": body})
        if response.get("body"):
            response["body"]["completion"] = response.get("output")

        if cache_response:
            rf = response.get("params", {}).get("response_format", {})
            if rf and not isinstance(rf, dict):
                response["params"]["response_format"] = {
                    "type": "json_schema",
                    "schema_name": getattr(rf, "__name__", str(rf)),
                }

            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=response,
                model_name=self.model.model_name,
                log_errors=log_errors,
            )

        return response

    async def _run_one(self, coro):
        async with self.semaphore:
            return await coro

    async def run_batch(self, coros: list):
        wrapped = [self._run_one(c) for c in coros]
        results = await asyncio.gather(*wrapped, return_exceptions=True)

        out = []
        for result in results:
            if isinstance(result, Exception):
                out.append({"error": str(result), "status": 500, "output": None, "usage": None})
            else:
                out.append(result)
        return out

    async def iter_batch(self, coros: list):
        for fut in asyncio.as_completed([self._run_one(c) for c in coros]):
            yield await fut


__all__ = ["OpenAIClient"]
