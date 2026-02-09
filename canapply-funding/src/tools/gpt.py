# src/tools/gpt.py

import asyncio
import base64
import json
import hashlib
import hmac
from src.config import settings
import six
from pathlib import Path
import time
import uuid
from typing import Union, ClassVar, Dict, Optional, Tuple

import aiohttp
import aiofiles
from blake3 import blake3
import tiktoken
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
import openai

class QdrantCache:
    """
    Payload-only cache in Qdrant.
    We keep a 1-D dummy vector so the collection can exist even if we never do ANN.
    Items are fetched by `identifier` + `client_type` filter.
    """
    def __init__(
        self,
        collection: str = "vdb_collection",
        client_type: str = "completions",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.QDRANT_URL).rstrip("/")
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection = collection
        self.client_type = client_type
        self._ensured = False
        self._ensure_lock = asyncio.Lock()

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    async def ensure_ready(self):
        await self._ensure_collection()

    async def _ensure_collection(self):
        async with self._ensure_lock:
            if self._ensured:
                return
            async with aiohttp.ClientSession() as s:
                url = f"{self.base_url}/collections/{self.collection}"
                async with s.get(url, headers=self._headers()) as r:
                    if r.status == 200:
                        self._ensured = True
                        return
                body = {"vectors": {"size": 1, "distance": "Dot"}}
                async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                    if r.status in (200, 201, 409):  # 409 == already created by someone else
                        self._ensured = True
                        return
                    txt = await r.text()
                    raise RuntimeError(f"Failed to create Qdrant collection: {r.status} {txt}")

    @staticmethod
    def _u64_hash(s: str) -> int:
        return int.from_bytes(blake3(s.encode("utf-8")).digest()[:8], "big", signed=False)

    async def _exists(self, identifier: str) -> bool:
        await self._ensure_collection()
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points/scroll"
            body = {
                "filter": {
                    "must": [
                        {"key": "identifier", "match": {"value": identifier}},
                        {"key": "client_type", "match": {"value": self.client_type}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return False
                data = await r.json()
                return bool(data.get("result", {}).get("points"))

    async def resolve_identifier(
        self, identifier: str, rewrite_cache: bool, regen_cache: bool
    ) -> Tuple[str, bool]:
        """
        Returns (effective_identifier, can_read_existing).
        We mimic file behavior:
          - completions + rewrite_cache => choose next unused "identifier_i" and don't read.
          - otherwise => use "identifier" and read if not regen.
        """
        if self.client_type == "completions" and rewrite_cache and not regen_cache:
            # find the next available suffix (0..999)
            for i in range(0, 1000):
                eff = f"{identifier}_{i}"
                if not await self._exists(eff):
                    return eff, False
            # fallback if all taken
            return f"{identifier}_{int(time.time())}", False
        # embeddings or normal completions path
        return identifier, (not regen_cache)

    async def read(self, effective_identifier: str) -> Optional[dict]:
        await self._ensure_collection()
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points/scroll"
            body = {
                "filter": {
                    "must": [
                        {"key": "identifier", "match": {"value": effective_identifier}},
                        {"key": "client_type", "match": {"value": self.client_type}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                pts = data.get("result", {}).get("points", [])
                if not pts:
                    return None
                payload = pts[0].get("payload", {})
                return payload.get("cache")

    async def write(
        self,
        effective_identifier: str,
        response: dict,
        model_name: str,
    ) -> None:
        await self._ensure_collection()
        payload = {
            "identifier": effective_identifier,
            "base_identifier": response.get("identifier"),
            "client_type": self.client_type,
            "model": model_name,
            "error": response.get("error"),
            "status": response.get("status"),
            "cache": response,  # full cached response blob
            "created_at": int(time.time()),
        }

        # NOTE: we store a dummy 1D vector; this collection is a payload cache.
        point = {
            "id": self._u64_hash(effective_identifier),  # deterministic numeric id
            "vector": [0.0],
            "payload": payload,
        }

        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points?wait=true"
            body = {"points": [point]}
            async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status not in (200, 202):
                    txt = await r.text()
                    print(f"[QdrantCache] upsert failed: {r.status} {txt}")


class ModelSpec:
    key: ClassVar[str]
    model_name: ClassVar[str]
    category: ClassVar[str]
    context_window: ClassVar[int]
    rate_limits: ClassVar[dict]
    usage_costs: ClassVar[dict]

    max_output: ClassVar[int | None] = None
    output_dimensions: ClassVar[int | None] = None

    _registry: ClassVar[Dict[str, "ModelSpec"]] = {}

    def __init_subclass__(cls, **kwargs):
        if getattr(cls, "key", None) in cls._registry:
            raise ValueError(f"Duplicate model key: {cls.key}")
        cls._registry[cls.key] = cls

    @classmethod
    def input_cost(cls, n_tokens: int) -> float:
        return n_tokens * cls.usage_costs["input"]

    @classmethod
    def output_cost(cls, n_tokens: int) -> float:
        if cls.category != "completions":
            return 0.0
        return n_tokens * cls.usage_costs["output"]

    @classmethod
    def get(cls, key: str) -> "ModelSpec":
        try:
            return cls._registry[key]
        except KeyError:
            raise ValueError(f"Unknown model key {key!r}") from None


class GPT5Point1(ModelSpec):
    key = "gpt-5.1"
    model_name = "gpt-5.1-2025-11-13"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}


class GPT5(ModelSpec):
    key = "gpt-5"
    model_name = "gpt-5-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}


class GPT5Mini(ModelSpec):
    key = "gpt-5-mini"
    model_name = "gpt-5-mini-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": 0.25 / 1_000_000,
        "output": 2.00 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}


class GPT5Nano(ModelSpec):
    key = "gpt-5-nano"
    model_name = "gpt-5-nano-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": 0.05 / 1_000_000,
        "output": 0.40 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}


class TextEmbedding3Large(ModelSpec):
    key = "text-embedding-3-large"
    model_name = "text-embedding-3-large"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 3_072
    usage_costs = {
        "input": 0.13 / 1_000_000
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}


class TextEmbedding3Small(ModelSpec):
    key = "text-embedding-3-small"
    model_name = "text-embedding-3-small"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {
        "input": 0.02 / 1_000_000
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}


class Tokenizer:
    def __init__(self, model_specs: dict = None) -> None:
        self._encoder = tiktoken.get_encoding("cl100k_base")
        if model_specs:
            self._usage_costs = model_specs["usage_costs"]

    def tokenizer(self, context: str):
        return self._encoder.encode(context)

    def count_tokens(self, context):
        try:
            if isinstance(context, str):
                return len(self.tokenizer(context))
            elif isinstance(context, list):
                if isinstance(context[0], str):
                    return len(self.tokenizer(context[0]))

                per_message = 4
                num_tokens = 0
                for message in context:
                    num_tokens += per_message
                    for key, value in message.items():
                        num_tokens += len(self.tokenizer(value))
                num_tokens += 3
                return num_tokens
            else:
                return len(self.tokenizer(str(context)))
        except Exception as e:
            print(f"Error While Counting Tokens: {e}")
            return 0

    def parse_usage(self, usage: dict) -> dict:
        if not self._usage_costs:
            raise ValueError("Usage costs not defined for the model. Pass the model_specs dictionary to the Tokenizer class")

        parsed_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        if "completion_tokens" in usage:
            parsed_usage["output_tokens"] = usage["completion_tokens"]
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"].get("cached_tokens", 0) > 0:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            input_cost_cached = cached_tokens * self._usage_costs["cached_input"]
            input_cost_uncached = (parsed_usage["input_tokens"] - cached_tokens) * self._usage_costs["input"]
            input_cost = input_cost_cached + input_cost_uncached
            parsed_usage["input_tokens_cached"] = cached_tokens
        else:
            input_cost = parsed_usage["input_tokens"] * self._usage_costs["input"]

        parsed_usage["input_cost"] = input_cost
        if "output_tokens" in parsed_usage:
            parsed_usage["output_cost"] = parsed_usage["output_tokens"] * self._usage_costs["output"]
        parsed_usage["total_cost"] = parsed_usage["input_cost"] + parsed_usage.get("output_cost", 0)

        return parsed_usage


class TokenBucket:
    def __init__(self, size: int = 0) -> None:
        self._maximum_size = size
        self._current_size = size
        self._consume_per_second = size / 60
        self._last_fill_time = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, amount: int = 0) -> None:
        if amount == 0:
            return

        async with self._lock:
            if amount > self._maximum_size:
                raise ValueError("Amount exceeds bucket size.")

            self._refill()

            while amount > self._current_size:
                await asyncio.sleep(0.05)
                self._refill()

            self._current_size -= amount

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_fill_time
        refilled_tokens = int(elapsed * self._consume_per_second)
        self._current_size = min(self._maximum_size, self._current_size + refilled_tokens)
        self._last_fill_time = now


class Limiter:
    def __init__(self, model_specs: dict = None) -> None:
        self.tkn_limiter = TokenBucket(size=int(model_specs["rate_limits"]["tkn_per_min"] * 0.75))
        self.req_limiter = TokenBucket(size=int(model_specs["rate_limits"]["req_per_min"] * 0.95))

    def limit(self, tokens: int = 0, requests: int = 0):
        return self._LimitContextManager(self, tokens, requests)

    class _LimitContextManager:
        def __init__(self, limiter, tokens, requests):
            self.limiter = limiter
            self.tokens = tokens
            self.requests = requests
            self.output_tokens = 0

        async def __aenter__(self):
            await asyncio.gather(
                self.limiter.tkn_limiter.consume(self.tokens),
                self.limiter.req_limiter.consume(self.requests)
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.output_tokens > 0:
                await self.limiter.tkn_limiter.consume(self.output_tokens)

class Streamer:
    def __init__(self, channel: str = None) -> None:
        self.auth_key = settings.PUSHER_AUTH_KEY
        self.auth_secret = (settings.PUSHER_AUTH_SECRET or "").encode('utf8')
        self.auth_version = settings.PUSHER_AUTH_VERSION

        self.base = f"https://api-{settings.PUSHER_APP_CLUSTER}.pusher.com"
        self.path = f"/apps/{settings.PUSHER_APP_ID}/events"
        self.headers = {
            "X-Pusher-Library": f"pusher-http-python {self.auth_version}",
            "Content-Type": "application/json"
        }

        self.channel = channel
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.session.close()

    def _generate_query_string(self, params: dict) -> str:
        body_md5 = hashlib.md5(json.dumps(params).encode('utf-8')).hexdigest()
        query_params: dict = {
            "auth_key": self.auth_key,
            "auth_timestamp": str(int(time.time())),
            "auth_version": self.auth_version,
            "body_md5": six.text_type(body_md5)
        }
        query_string = '&'.join(map('='.join, sorted(query_params.items(), key=lambda x: x[0])))
        auth_string = '\n'.join(["POST", self.path, query_string]).encode('utf8')
        signature_encoded = hmac.new(self.auth_secret, auth_string, hashlib.sha256).hexdigest()
        query_params["auth_signature"] = six.text_type(signature_encoded)
        query_string += "&auth_signature=" + query_params["auth_signature"]

        return query_string

    async def push_event(self, name: str, data: str) -> Union[dict, str]:
        params = {"name": name, "data": data, "channels": [self.channel]}
        query_string = self._generate_query_string(params)
        url = f"{self.base}{self.path}?{query_string}"
        data = json.dumps(params)

        try:
            async with self.session.post(url=url, data=data, headers=self.headers) as response:
                if response.headers.get("Content-Type") == "application/json":
                    return await response.json()
                return await response.text()
        except aiohttp.ClientError as e:
            return str(e)


class ResponseTimeoutError(asyncio.TimeoutError):
    """Raised when get_response exceeds the user‑supplied timeout."""

class OpenAIClient:
    model_support = {
        "completions": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1"
        ],
        "embeddings": [
            "text-embedding-3-large",
            "text-embedding-3-small",
        ]
    }

    def __init__(
        self,
        model_name: str,
        cache_dir: Union[str, Path, None] = None,
        *,
        vdb_toggle: bool = False,
        vdb_collection: str = None,
        vdb_url: str | None = None,
        vdb_api_key: str | None = None,
    ) -> None:
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.vdb_toggle = bool(vdb_toggle)

        self.cache_backlogs = bool(cache_dir) or self.vdb_toggle
        if (self.cache_dir and not self.vdb_toggle) and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.spec = ModelSpec.get(model_name)
        self._client_type = self.spec.category
        if model_name in self.model_support["completions"]:
            self._call_model = self._call_completions
            self._client_type = "completions"
        elif model_name in self.model_support["embeddings"]:
            self._call_model = self._call_embeddings
            self._client_type = "embeddings"
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.model_name = self.spec.model_name
        self.tokenizer = Tokenizer(vars(self.spec))
        self.limiter = Limiter(vars(self.spec))
        self.openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.semaphore = asyncio.Semaphore(256)

        self.vdb_client = None
        if self.vdb_toggle:
            if not vdb_collection:
                vdb_collection = self._client_type + "_cache"
            self.vdb_client = QdrantCache(
                collection=vdb_collection,
                client_type=self._client_type,
                base_url=vdb_url,
                api_key=vdb_api_key,
            )

    async def warm_cache(self) -> None:
        if self.vdb_client and hasattr(self.vdb_client, "preload_all"):
            await self.vdb_client.preload_all()

    def is_cached(self, identifier: str) -> bool:
        if self.vdb_client and getattr(self.vdb_client, "_cache_loaded", False):
            return identifier in self.vdb_client._cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{identifier}.json"
            return cache_path.exists()
        return False

    async def _call_completions(self, **params) -> dict:
        assert params.get("messages"), "Messages are required for completions."
        params["model"] = self.model_name

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

        async with self.limiter.limit(tokens=self.tokenizer.count_tokens(params["messages"]), requests=1) as limit_context:
            try:
                if params.get("stream", False):
                    output = ""
                    params["stream_options"] = {"include_usage": True}
                    async with Streamer(channel=params.pop("channel", str(uuid.uuid4()))) as streamer:
                        listener = await self.openai.chat.completions.create(**params)
                        await streamer.push_event('new-response', "")
                        async for chunk in listener:
                            if not chunk.choices and chunk.usage:
                                usage = dict(chunk.usage)
                                usage["completion_tokens_details"] = usage["completion_tokens_details"].dict()
                                usage["prompt_tokens_details"] = usage["prompt_tokens_details"].dict()
                                error, status = "OK", 200
                                break
                            token = chunk.choices[0].delta.content
                            if token is not None:
                                await streamer.push_event('new-token', token)
                                output += token
                        await streamer.push_event('response-finished', error)
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

                usage = self.tokenizer.parse_usage(usage)
                limit_context.output_tokens = usage["output_tokens"]
            except openai.APIConnectionError as e:
                status, error = 500, e.__cause__
                print(f"API Connection Error in Completions: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Completions: {error}")
            except openai.APIStatusError as e:
                status, error = e.status_code, e.response
                print(f"API Status Error in Completions: {error}")
            finally:
                usage = usage or {}
        return dict(params=params, output=output, usage=usage,
                            status=status, error=error)

    async def _call_embeddings(self, **params) -> dict:
        assert params.get("input"), "Input is required for embeddings."
        params["model"] = self.model_name
        if not params.get("encoding_format"):
            params["encoding_format"] = "base64"

        output, usage = None, {}
        status, error = 500, "INCOMPLETE"

        input_tokens = self.tokenizer.count_tokens(params["input"])

        async with self.limiter.limit(tokens=input_tokens, requests=1):
            try:
                response = await self.openai.embeddings.create(**params)
                output = [d.embedding for d in response.data]
                if params["encoding_format"] == "base64":
                    output = [
                        np.frombuffer(base64.b64decode(o), dtype=np.float32).tolist()
                        for o in output if isinstance(o, str)
                    ]
                output = output[0] if len(output) == 1 else output
                usage = self.tokenizer.parse_usage(response.usage.to_dict())
                status, error = 200, "OK"
            except openai.APIConnectionError as e:
                status, error = 500, e.__cause__
                print(f"API Connection Error in Embeddings: {error}")
            except openai.RateLimitError:
                status, error = 429, "Rate limit exceeded."
                print(f"Rate Limit Error in Embeddings: {error}")
            except openai.APIStatusError as e:
                status, error = e.status_code, e.response
                print(f"API Status Error in Embeddings: {error}")
            finally:
                usage = usage or {}
        return dict(params=params, output=output, usage=usage,
                            status=status, error=error)

    def _get_cache_path(self, identifier: str, rewrite_cache: bool, regen_cache: bool) -> Path:
        cache_path = None
        if self.cache_backlogs:
            if self._client_type == "completions":
                if not rewrite_cache or regen_cache:
                    cache_path = self.cache_dir / f"{identifier}.json"
                else:
                    for i in range(0, 1000, 1):
                        cache_path = self.cache_dir / f"{identifier}_{i}.json"
                        if not cache_path.exists():
                            break
            else:
                cache_path = self.cache_dir / f"{identifier}.json"
        return cache_path

    async def get_response(
            self,
            identifier: str = None,
            attempts: int = 3,
            backoff: int = 1,
            body: dict = None,
            cache_response: bool = False,
            rewrite_cache: bool = False,
            regen_cache: bool = False,
            log_errors: bool = True,
            timeout: float | None = None,
            **kwargs
    ) -> dict:
        identifier = identifier or blake3(str(uuid.uuid4()).encode('utf-8')).hexdigest()

        cache_path = None
        if cache_response and self.cache_backlogs:
            if self.vdb_toggle and self.vdb_client:
                eff_id, can_read = await self.vdb_client.resolve_identifier(identifier, rewrite_cache, regen_cache)
                if can_read:
                    resp = await self.vdb_client.read(eff_id)
                    if resp and resp.get("error") == "OK":
                        return resp
            else:
                cache_path = self._get_cache_path(identifier, rewrite_cache, regen_cache)
                if cache_path and cache_path.exists() and not regen_cache:
                    resp = await self._read_json(cache_path)
                    if resp.get("error") == "OK":
                        return resp

        response = {}
        while attempts > 0:
            try:
                if timeout is None:
                    response = await self._call_model(**kwargs)
                else:
                    response = await asyncio.wait_for(
                        self._call_model(**kwargs),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                attempts -= 1
                if attempts == 0:
                    return {"status": 408, "error": "Timeout", "output": None, "usage": {}}
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            if response["status"] < 500:
                break

            attempts -= 1
            if attempts == 0:
                return response

            await asyncio.sleep(backoff)
            backoff *= 2

        response.update({"identifier": identifier, "body": body})
        if response["body"]:
            response["body"]["completion"] = response["output"]

        if cache_response and self.cache_backlogs:
            if not isinstance(response.get("params", {}).get("response_format", {}), dict):
                response["params"]["response_format"] = {
                    "type": "json_schema",
                    "schema_name": response["params"]["response_format"].__name__
                }

            if self.vdb_toggle and self.vdb_client:
                eff_id, _ = await self.vdb_client.resolve_identifier(identifier, rewrite_cache, regen_cache)
                if response["error"] == "OK" or (response["error"] != "OK" and log_errors):
                    await self.vdb_client.write(eff_id, response, model_name=self.model_name)
            else:
                if response["error"] == "OK" or (response["error"] != "OK" and log_errors):
                    await self._write_json(cache_path, response)

        return response

    async def _run_one(self, coro):
        async with self.semaphore:
            return await coro

    async def run_batch(self, coros: list):
        wrapped = [self._run_one(c) for c in coros]
        results = await asyncio.gather(*wrapped, return_exceptions=True)

        out = []
        for r in results:
            if isinstance(r, Exception):
                out.append({"error": str(r), "status": 500, "output": None, "usage": None})
            else:
                out.append(r)
        return out

    async def iter_batch(self, coros: list):
        for fut in asyncio.as_completed([self._run_one(c) for c in coros]):
            yield await fut

    @staticmethod
    async def _read_json(path: str | Path, default: dict | list = None) -> dict | list:
        try:
            async with aiofiles.open(path, "r") as file:
                return json.loads(await file.read())
        except (FileNotFoundError, json.JSONDecodeError):
            default = {} if default is None else default
        return default

    @staticmethod
    async def _write_json(path: str | Path, data: dict | list, indent: int = 4, encoding: str = "utf-8") -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == ".json":
            path = path.with_suffix(".json")

        if not data or data.get("error") != "OK":
            path = path.with_name(f"{path.stem}_error{path.suffix}")
            print(f"Writing error log to {path}")

        try:
            async with aiofiles.open(path, "w", encoding=encoding) as file:
                await file.write(json.dumps(data, indent=indent))
        except Exception as e:
            print(f"Error Writing JSON: {e}")

    async def clean_up_cache_directory(self):
        if not self.cache_backlogs:
            print("Cache directory not set up.")
            return

        for file_path in tqdm(self.cache_dir.glob("*.json"), desc="Cleaning cache directory"):
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = json.loads(await f.read())
                if content.get("error") != "OK":
                    print(f"Deleting incomplete cache file: {file_path}")
                    file_path.unlink()
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Deleting invalid cache file: {file_path}")
                file_path.unlink()


async def main():
    s = time.perf_counter()
    client = OpenAIClient(
        "gpt-5-nano",
        vdb_toggle=True,
        vdb_collection="completions",
        vdb_url="http://localhost:6333"
    )
    resp = await client.get_response(
        identifier="test-completions",
        cache_response=True,
        messages=[{"role": "user", "content": "Hello world"}],
    )
    elapsed = time.perf_counter() - s

    print(f"Elapsed Time: {elapsed:0.2f} seconds")
    print(f"Response: {resp["output"]}")


if __name__ == "__main__":
    asyncio.run(main())
