"""
Backward-compatible OpenAI client wrapper.

This module provides the original OpenAIClient interface while delegating
to the new provider architecture. Existing code using OpenAIClient will
continue to work unchanged but will now benefit from the ExecutionEngine's
robustness (circuit breakers, hooks, etc).

For new code, consider using the provider and agent APIs directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
import warnings
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3

from .cache import CacheSettings, build_cache_core
from .config import get_settings
from .engine import ExecutionEngine, RetryConfig
from .models import ModelProfile
from .providers.openai import OpenAIProvider
from .providers.types import CompletionResult, StreamEventType, normalize_messages
from .spec import RequestSpec
from .streaming import PusherStreamer, format_sse_event

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    High-level OpenAI client wrapper.

    Acts as a facade over the ExecutionEngine.
    """

    def __init__(
        self,
        model: type[ModelProfile] | str | None = None,
        *,
        cache_dir: str | Path | None = None,
        responses_api_toggle: bool = False,
        use_engine: bool = True,  # Defaults to True now
        engine: ExecutionEngine | None = None,
        cache_backend: Literal["qdrant", "pg_redis", "fs"] | None = None,
        cache_collection: str | None = None,
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

        # Load global settings if available or defaults
        self.settings = get_settings()

        # Model setup
        if isinstance(model, type) and issubclass(model, ModelProfile):
            self.model = model
        elif isinstance(model, str):
            self.model = ModelProfile.get(model)
        elif model is None:
            # Fallback to configured default
            self.model = ModelProfile.get(self.settings.openai.default_model)
        else:
            raise ValueError("Model must be a ModelProfile usage or key string.")

        # Cache setup
        backend_name = cache_backend or "none"
        self.cache = build_cache_core(
            CacheSettings(
                backend=backend_name,
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

        # Engine setup
        self.engine: ExecutionEngine
        if engine:
            self.engine = engine
        else:
            # Create our own engine with OpenAIProvider
            provider = OpenAIProvider(
                model=self.model,
                use_responses_api=self.responses_api_toggle,
                cache_backend=None if self.cache else None,  # We handle caching at engine level if we pass cache object
            )
            self.engine = ExecutionEngine(
                provider=provider,
                cache=self.cache,
                max_concurrency=self.settings.agent.batch_concurrency,
            )

        # Metrics setup
        if self.settings.metrics.enabled:
            from .hooks import OpenTelemetryHook, PrometheusHook

            provider_type = self.settings.metrics.provider
            if provider_type == "prometheus":
                try:
                    self.engine.hooks.add(PrometheusHook(port=self.settings.metrics.prometheus_port))
                    logger.info("Enabled Prometheus metrics on port %s", self.settings.metrics.prometheus_port)
                except ImportError:
                    logger.warning("prometheus_client not installed, metrics disabled.")
            elif provider_type == "otel":
                try:
                    self.engine.hooks.add(OpenTelemetryHook())
                    logger.info("Enabled OpenTelemetry metrics")
                except ImportError:
                    logger.warning("opentelemetry not installed, metrics disabled.")

        # Deprecation warnings for old flags
        if not use_engine:
            warnings.warn(
                "The 'use_engine=False' flag is deprecated and ignored. "
                "OpenAIClient now always uses the ExecutionEngine.",
                DeprecationWarning,
                stacklevel=2,
            )

    async def warm_cache(self) -> None:
        """Pre-warm the cache."""
        if self.cache:
            await self.cache.warm()

    async def batch(
        self,
        specs: Iterable[RequestSpec],
        **kwargs: Any,
    ) -> list[CompletionResult]:
        """
        Execute a batch of requests concurrently.

        Args:
            specs: List of request specifications
            **kwargs: Arguments passed to engine.batch_complete()

        Returns:
            List of CompletionResults
        """
        return await self.engine.batch_complete(specs, **kwargs)

    async def close(self) -> None:
        """Close client resources."""
        if self.cache:
            await self.cache.close()
        # Engine checks?
        # self.engine.provider.close() if needed

    @staticmethod
    def encode_file(file_path: str | Path) -> dict[str, Any]:
        """Encode a file for API submission."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file:
            data = file.read()
        base64_encoded = base64.b64encode(data).decode("utf-8")

        if extension in {".jpg", ".jpeg", ".png", ".webp"}:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/{extension[1:]};base64,{base64_encoded}"},
            }
        elif extension == ".pdf":
            return {
                "type": "input_file",
                "filename": file_path.name,
                "file_data": base64_encoded,
            }
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    async def transcribe_pdf(self, file_path: str | Path) -> dict[str, Any]:
        """Extract text from a PDF file."""
        warnings.warn(
            "transcribe_pdf bypasses ExecutionEngine and lacks caching/retry/hooks. "
            "Consider using OpenAIProvider with ExecutionEngine for production use. "
            "This method will be moved to a separate utility in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        file_path = Path(file_path)
        # Note: This relies on OpenAI's beta features or specific provider support
        # We construct a message compatible with the provider's expectations

        # NOTE: Logic duplicated from original client, but cleaner
        # Original logic used OpenAI 'Upload File' API first.
        # We will assume the file encoding helper is sufficient or we need to
        # replicate the file-upload logic if the provider requires file_ids.

        # Actually checking old logic: it DID call self.openai.files.create()
        # The engine provider has its own client instance, distinct from ours unless shared.
        # This is a leaky abstraction issue.
        # For now, we will retain the direct file upload here since 'ExecutionEngine' likely
        # doesn't expose file management API yet.

        # We access the provider's inner client if available, or create a temp one?
        # Best approach: Use the provider's client if it exposes it, or just use `openai` lib directly here.
        from openai import AsyncOpenAI

        # We need API key from settings
        aclient = AsyncOpenAI(api_key=self.settings.openai.api_key)

        try:
            with open(file_path, "rb") as f:
                file_obj = await aclient.files.create(file=f, purpose="user_data")
        finally:
            await aclient.close()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract text from this PDF document. Raw text only."},
                    {"type": "input_file", "file_id": file_obj.id},
                ],
            }
        ]

        # Use responses API (beta) via engine
        result = await self.get_response(input=messages, reasoning={"effort": "low"}, stream=False)
        if not isinstance(result, dict):
            raise TypeError("Expected a dict response")
        return result

    async def transcribe_image(self, file_path: str | Path) -> dict[str, Any]:
        """Extract text from an image file."""
        warnings.warn(
            "transcribe_image bypasses some ExecutionEngine features. "
            "Consider using provider.complete() directly for full feature support.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use existing logic helper
        file_path = Path(file_path)
        with open(file_path, "rb") as file:
            data = file.read()
        base64_encoded = base64.b64encode(data).decode("utf-8")
        ext = file_path.suffix[1:]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe text from image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{base64_encoded}"}},
                ],
            }
        ]
        result = await self.get_response(messages=messages, response_format="text", stream=False)
        if not isinstance(result, dict):
            raise TypeError("Expected a dict response")
        return result

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
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Uniifed entry point for completions and embeddings."""

        # 1. Detect if this is an embedding request
        if self.model.category == "embeddings":
            return await self._handle_embeddings(kwargs.get("input"), kwargs, timeout=timeout)

        # 2. Extract standard arguments
        params = dict(kwargs)
        messages = params.pop("messages", None)
        # Fallback for "input" which is sometimes used (e.g. Responses API)
        if messages is None:
            messages = params.pop("input", None)
        if messages is None:
            # Legacy: 'prompt' might be used
            messages = params.pop("prompt", "")

        tool_choice = params.pop("tool_choice", None)
        tools = params.pop("tools", None)
        temperature = params.pop("temperature", None)
        max_tokens = params.pop("max_tokens", None)
        response_format = params.pop("response_format", None)
        reasoning_effort = params.pop("reasoning_effort", None)
        reasoning = params.pop("reasoning", None)

        stream = bool(params.pop("stream", False))
        stream_mode = params.pop("stream_mode", "pusher")
        channel = params.pop("channel", None)

        # 3. Build Retry Configuration
        retry_cfg = RetryConfig(
            attempts=attempts or 3,
            backoff=backoff or 1.0,
        )

        # 4. Create RequestSpec
        msg_objects = normalize_messages(messages)
        spec = RequestSpec(
            provider="OpenAIProvider",  # Could be dynamic if we supported switch
            model=self.model.model_name,
            messages=msg_objects,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            extra=params,
            stream=stream,
        )

        # 5. Determine Cache Identifier
        if not identifier:
            if hash_as_identifier:
                identifier = spec.cache_key()
            else:
                identifier = blake3(str(uuid.uuid4()).encode("utf-8")).hexdigest()

        # 6. Execute via Engine

        # STREAMING PATH
        if stream:
            return await self._handle_streaming(spec, stream_mode, channel, self.engine)

        # COMPLETION PATH
        async def _do_complete():
            return await self.engine.complete(
                spec,
                cache_response=cache_response,
                cache_collection=cache_collection or self.default_cache_collection,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                cache_key=identifier if cache_response else None,
                retry=retry_cfg,
            )

        if timeout is None:
            result_obj = await _do_complete()
        else:
            result_obj = await asyncio.wait_for(_do_complete(), timeout=timeout)

        # 7. Convert Result to Legacy Dict Format

        # Prepare body meta
        response_body = body or {}
        if result_obj.content:
            response_body["completion"] = result_obj.content

        final_result = {
            "params": spec.to_dict(),
            "output": result_obj.content,
            "usage": result_obj.usage.to_dict() if result_obj.usage else {},
            "status": result_obj.status,
            "error": result_obj.error or "OK",
            "identifier": identifier,
            "body": response_body,
        }

        if return_response:
            final_result["response"] = result_obj.raw_response

        return final_result

    async def _handle_streaming(
        self,
        spec: RequestSpec,
        mode: str,
        channel: str | None,
        engine: ExecutionEngine,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Handle streaming responses (SSE or Pusher)."""
        if mode == "sse":
            # SSE mode returns an async iterator of SSE-formatted strings.
            async def sse_gen():
                output = ""
                usage = {}
                yield format_sse_event("meta", json.dumps({"model": spec.model, "stream_mode": "sse"}))

                async for event in engine.stream(spec):
                    if event.type == StreamEventType.TOKEN:
                        output += event.data
                        yield format_sse_event("token", event.data)
                    elif event.type == StreamEventType.USAGE:
                        usage = event.data.to_dict() if hasattr(event.data, "to_dict") else event.data
                    elif event.type == StreamEventType.ERROR:
                        yield format_sse_event("error", json.dumps(event.data))
                    elif event.type == StreamEventType.DONE:
                        # Final event
                        if event.data and event.data.usage:
                            usage = event.data.usage.to_dict()
                        yield format_sse_event(
                            "done",
                            json.dumps({"status": 200, "usage": usage, "output": output}),
                        )

            return sse_gen()

        # Pusher mode returns an awaitable that resolves to a dict.
        async def run_pusher() -> dict[str, Any]:
            output = ""
            usage: dict[str, Any] = {}
            status = 200
            error = "OK"
            channel_id = channel or str(uuid.uuid4())

            async with PusherStreamer(channel=channel_id) as streamer:
                await streamer.push_event("new-response", "")

                async for event in engine.stream(spec):
                    if event.type == StreamEventType.TOKEN:
                        await streamer.push_event("new-token", event.data)
                        output += event.data
                    elif event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
                        usage = event.data.to_dict()
                    elif event.type == StreamEventType.ERROR:
                        status = int(event.data.get("status", 500)) if isinstance(event.data, dict) else 500
                        error = str(event.data.get("error")) if isinstance(event.data, dict) else str(event.data)
                        await streamer.push_event(
                            "error",
                            json.dumps(event.data) if isinstance(event.data, dict) else str(event.data),
                        )
                    elif event.type == StreamEventType.DONE:
                        if event.data and event.data.usage:
                            usage = event.data.usage.to_dict()

                await streamer.push_event("response-finished", error)

            return {
                "channel": channel_id,
                "output": output,
                "usage": usage,
                "status": status,
                "error": error,
            }

        return await run_pusher()

    async def _handle_embeddings(
        self,
        inputs: Any,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Handle embedding request via engine."""
        # Using ExecutionEngine.embed() which calls provider.embed() and returns an EmbeddingResult.
        embed_params = dict(params)
        embed_params.pop("messages", None)
        embed_params.pop("prompt", None)
        embed_params.pop("stream", None)
        embed_params.pop("stream_mode", None)
        embed_params.pop("channel", None)
        embed_params.pop("input", None)

        try:
            result = await self.engine.embed(inputs, timeout=timeout, **embed_params)
        except Exception as e:
            return {"status": 500, "error": str(e), "output": [], "usage": {}}

        if hasattr(result, "embeddings"):
            output = result.embeddings
            usage = result.usage.to_dict() if getattr(result, "usage", None) else {}
            return {
                "output": output,
                "usage": usage,
                "status": getattr(result, "status", 200),
                "error": getattr(result, "error", None) or "OK",
                "model": getattr(result, "model", None),
            }

        # Fallback for unexpected return types.
        return {
            "output": result,
            "usage": {},
            "status": 200,
            "error": "OK",
        }

    # =========================================================================
    # Deprecated / Legacy Methods
    # =========================================================================

    def _call_model(self, **kwargs):
        warnings.warn("Use get_response() instead.", DeprecationWarning, stacklevel=2)
        pass  # Replaced by logic in get_response

    async def _call_completions(self, **kwargs):
        warnings.warn("Legacy method. Use get_response().", DeprecationWarning, stacklevel=2)
        return await self.get_response(**kwargs)

    async def _call_responses(self, **kwargs):
        warnings.warn("Legacy method. Use get_response().", DeprecationWarning, stacklevel=2)
        # Map 'input' to 'messages' if needed, but get_response handles it
        return await self.get_response(**kwargs)

    async def _call_embeddings(self, **kwargs):
        warnings.warn("Legacy method. Use get_response().", DeprecationWarning, stacklevel=2)
        return await self.get_response(**kwargs)
