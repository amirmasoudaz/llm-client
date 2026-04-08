from __future__ import annotations

import asyncio

import pytest

from llm_client.cancellation import CancellationToken, CancelledError
from llm_client.engine import ExecutionEngine, FailoverPolicy, RetryConfig
from llm_client.hooks import HookManager
from llm_client.idempotency import IdempotencyTracker
from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.tools import ResponsesAttributeFilter, ResponsesFileSearchRankingOptions
from llm_client.providers.types import (
    AudioSpeechResult,
    AudioTranscriptionResult,
    BackgroundResponseResult,
    CompactionResult,
    CompletionResult,
    DeepResearchRunResult,
    ConversationItemResource,
    ConversationItemsPage,
    ConversationResource,
    DeletionResult,
    FileContentResult,
    FileResource,
    FilesPage,
    FineTuningJobEventsPage,
    FineTuningJobResult,
    GeneratedImage,
    ImageGenerationResult,
    Message,
    ModerationResult,
    RealtimeCallResult,
    RealtimeConnection,
    RealtimeClientSecretResult,
    RealtimeTranscriptionSessionResult,
    StreamEvent,
    StreamEventType,
    Usage,
    VectorStoreFileBatchResource,
    VectorStoreFileContentResult,
    VectorStoreFileResource,
    VectorStoreFilesPage,
    VectorStoreResource,
    VectorStoreSearchResult,
    WebhookEventResult,
)
from llm_client.resilience import CircuitBreakerConfig
from llm_client.routing import RegistryRouter, StaticRouter
from llm_client.spec import RequestContext, RequestSpec
from tests.llm_client.fakes import ScriptedProvider, error_result, ok_result


def _spec() -> RequestSpec:
    return RequestSpec(
        provider="openai",
        model="gpt-5-mini",
        messages=[Message.user("hello")],
    )


class _CollectingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, event: str, payload: dict, context: RequestContext) -> None:
        self.events.append((event, dict(payload)))


class _SlowProvider(ScriptedProvider):
    def __init__(self, *, delay: float, model_name: str = "gpt-5-mini") -> None:
        super().__init__(model_name=model_name)
        self._delay = delay

    async def complete(self, messages, **kwargs):
        _ = (messages, kwargs)
        await asyncio.sleep(self._delay)
        return ok_result("slow-ok", model=self.model_name)

    async def stream(self, messages, **kwargs):
        _ = (messages, kwargs)
        await asyncio.sleep(self._delay)
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=ok_result("slow-ok", model=self.model_name),
        )


class _FakeRealtimeSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    async def send(self, event) -> None:
        self.sent.append(event)

    async def recv(self):
        return {"type": "session.updated"}

    async def recv_bytes(self) -> bytes:
        return b"abc"

    async def close(self) -> None:
        self.closed = True


class _WorkflowProvider(ScriptedProvider):
    def __init__(self, *, model_name: str = "gpt-5-mini") -> None:
        super().__init__(model_name=model_name)
        self.workflow_calls: list[tuple[str, dict[str, object]]] = []

    async def retrieve_background_response(self, response_id: str, **kwargs):
        self.workflow_calls.append(("retrieve_background_response", {"response_id": response_id, **kwargs}))
        return BackgroundResponseResult(response_id=response_id, lifecycle_status="completed", completion=ok_result("done", model=self.model_name))

    async def wait_background_response(self, response_id: str, *, poll_interval: float = 2.0, timeout: float | None = None, **kwargs):
        self.workflow_calls.append(
            (
                "wait_background_response",
                {"response_id": response_id, "poll_interval": poll_interval, "timeout": timeout, **kwargs},
            )
        )
        return BackgroundResponseResult(response_id=response_id, lifecycle_status="completed", completion=ok_result("done", model=self.model_name))

    async def cancel_background_response(self, response_id: str, **kwargs):
        self.workflow_calls.append(("cancel_background_response", {"response_id": response_id, **kwargs}))
        return BackgroundResponseResult(response_id=response_id, lifecycle_status="cancelled")

    async def create_conversation(self, *, items=None, metadata=None, **kwargs):
        self.workflow_calls.append(("create_conversation", {"items": items, "metadata": metadata, **kwargs}))
        return ConversationResource(conversation_id="conv_1", metadata=metadata)

    async def create_conversation_items(self, conversation_id: str, *, items, include=None, **kwargs):
        self.workflow_calls.append(
            (
                "create_conversation_items",
                {"conversation_id": conversation_id, "items": items, "include": include, **kwargs},
            )
        )
        return ConversationItemsPage(
            items=[ConversationItemResource(item_id="item_1", item_type="message", content="hello")],
            first_id="item_1",
            last_id="item_1",
            has_more=False,
        )

    async def list_conversation_items(self, conversation_id: str, *, after=None, include=None, limit=None, order=None, **kwargs):
        self.workflow_calls.append(
            (
                "list_conversation_items",
                {
                    "conversation_id": conversation_id,
                    "after": after,
                    "include": include,
                    "limit": limit,
                    "order": order,
                    **kwargs,
                },
            )
        )
        return ConversationItemsPage(
            items=[ConversationItemResource(item_id="item_2", item_type="function_call")],
            first_id="item_2",
            last_id="item_2",
            has_more=True,
        )

    async def retrieve_conversation_item(self, conversation_id: str, item_id: str, *, include=None, **kwargs):
        self.workflow_calls.append(
            (
                "retrieve_conversation_item",
                {"conversation_id": conversation_id, "item_id": item_id, "include": include, **kwargs},
            )
        )
        return ConversationItemResource(item_id=item_id, item_type="mcp_approval_response")

    async def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs):
        self.workflow_calls.append(
            ("delete_conversation_item", {"conversation_id": conversation_id, "item_id": item_id, **kwargs})
        )
        return ConversationResource(conversation_id=conversation_id)

    async def compact_response_context(self, *, messages=None, model=None, instructions=None, previous_response_id=None, **kwargs):
        self.workflow_calls.append(
            (
                "compact_response_context",
                {
                    "messages": messages,
                    "model": model,
                    "instructions": instructions,
                    "previous_response_id": previous_response_id,
                    **kwargs,
                },
            )
        )
        return CompactionResult(compaction_id="cmp_1")

    async def submit_mcp_approval_response(self, *, previous_response_id: str, approval_request_id: str, approve: bool, tools=None, **kwargs):
        self.workflow_calls.append(
            (
                "submit_mcp_approval_response",
                {
                    "previous_response_id": previous_response_id,
                    "approval_request_id": approval_request_id,
                    "approve": approve,
                    "tools": tools,
                    **kwargs,
                },
            )
        )
        return ok_result("approved", model=self.model_name)

    async def delete_response(self, response_id: str, **kwargs):
        self.workflow_calls.append(("delete_response", {"response_id": response_id, **kwargs}))
        return DeletionResult(resource_id=response_id)

    async def stream_background_response(self, response_id: str, *, starting_after: int | None = None, **kwargs):
        self.workflow_calls.append(
            ("stream_background_response", {"response_id": response_id, "starting_after": starting_after, **kwargs})
        )
        yield StreamEvent(type=StreamEventType.TOKEN, data="background ")
        yield StreamEvent(type=StreamEventType.DONE, data=ok_result("background done", model=self.model_name))

    async def moderate(self, inputs, **kwargs):
        self.workflow_calls.append(("moderate", {"inputs": inputs, **kwargs}))
        return ModerationResult(flagged=True, model=self.model_name, results=[{"flagged": True}])

    async def generate_image(self, prompt: str, **kwargs):
        self.workflow_calls.append(("generate_image", {"prompt": prompt, **kwargs}))
        return ImageGenerationResult(images=[GeneratedImage(url="https://example.com/image.png")], model=self.model_name)

    async def transcribe_audio(self, file, **kwargs):
        self.workflow_calls.append(("transcribe_audio", {"file": file, **kwargs}))
        return AudioTranscriptionResult(text="hello", language="en", model=self.model_name)

    async def synthesize_speech(self, text: str, *, voice: str, **kwargs):
        self.workflow_calls.append(("synthesize_speech", {"text": text, "voice": voice, **kwargs}))
        return AudioSpeechResult(audio=b"abc", format="mp3", model=self.model_name)

    async def create_file(self, *, file, purpose: str, **kwargs):
        self.workflow_calls.append(("create_file", {"file": file, "purpose": purpose, **kwargs}))
        return FileResource(file_id="file_1", filename="guide.pdf", purpose=purpose)

    async def retrieve_file(self, file_id: str, **kwargs):
        self.workflow_calls.append(("retrieve_file", {"file_id": file_id, **kwargs}))
        return FileResource(file_id=file_id, filename="guide.pdf", purpose="assistants")

    async def list_files(self, **kwargs):
        self.workflow_calls.append(("list_files", dict(kwargs)))
        return FilesPage(items=[FileResource(file_id="file_1", filename="guide.pdf", purpose="assistants")])

    async def delete_file(self, file_id: str, **kwargs):
        self.workflow_calls.append(("delete_file", {"file_id": file_id, **kwargs}))
        return DeletionResult(resource_id=file_id, deleted=True)

    async def get_file_content(self, file_id: str, **kwargs):
        self.workflow_calls.append(("get_file_content", {"file_id": file_id, **kwargs}))
        return FileContentResult(file_id=file_id, content=b"abc", media_type="application/pdf")

    async def create_vector_store(self, **kwargs):
        self.workflow_calls.append(("create_vector_store", dict(kwargs)))
        return VectorStoreResource(vector_store_id="vs_1", name="Docs")

    async def search_vector_store(self, vector_store_id: str, *, query, **kwargs):
        self.workflow_calls.append(("search_vector_store", {"vector_store_id": vector_store_id, "query": query, **kwargs}))
        return VectorStoreSearchResult(vector_store_id=vector_store_id, query=query, results=[{"file_id": "file_1"}])

    async def create_fine_tuning_job(self, **kwargs):
        self.workflow_calls.append(("create_fine_tuning_job", dict(kwargs)))
        return FineTuningJobResult(job_id="ftjob_1", status="queued", base_model="gpt-4o-mini")

    async def list_fine_tuning_events(self, job_id: str, **kwargs):
        self.workflow_calls.append(("list_fine_tuning_events", {"job_id": job_id, **kwargs}))
        return FineTuningJobEventsPage(job_id=job_id, events=[{"id": "ftevent_1", "message": "queued"}])

    async def create_realtime_client_secret(self, **kwargs):
        self.workflow_calls.append(("create_realtime_client_secret", dict(kwargs)))
        return RealtimeClientSecretResult(value="secret_1")

    async def connect_realtime(self, **kwargs):
        self.workflow_calls.append(("connect_realtime", dict(kwargs)))
        return RealtimeConnection(_FakeRealtimeSocket(), model=str(kwargs.get("model") or self.model_name), call_id="rtc_live_1")

    async def create_realtime_transcription_session(self, **kwargs):
        self.workflow_calls.append(("create_realtime_transcription_session", dict(kwargs)))
        return RealtimeTranscriptionSessionResult(value="tx_secret_1")

    async def connect_realtime_transcription(self, **kwargs):
        self.workflow_calls.append(("connect_realtime_transcription", dict(kwargs)))
        return RealtimeConnection(_FakeRealtimeSocket(), model=str(kwargs.get("model") or self.model_name), call_id="rtc_tx_1")

    async def create_realtime_call(self, sdp: str, **kwargs):
        self.workflow_calls.append(("create_realtime_call", {"sdp": sdp, **kwargs}))
        return RealtimeCallResult(call_id="rtc_1", sdp="answer", action="create")

    async def accept_realtime_call(self, call_id: str, **kwargs):
        self.workflow_calls.append(("accept_realtime_call", {"call_id": call_id, **kwargs}))
        return RealtimeCallResult(call_id=call_id, action="accept")

    async def unwrap_webhook(self, payload, headers, *, secret=None):
        self.workflow_calls.append(("unwrap_webhook", {"payload": payload, "headers": headers, "secret": secret}))
        return WebhookEventResult(event_id="wh_1", event_type="realtime.call.incoming", data={"call_id": "rtc_1"})

    async def verify_webhook_signature(self, payload, headers, *, secret=None, tolerance: int = 300):
        self.workflow_calls.append(
            ("verify_webhook_signature", {"payload": payload, "headers": headers, "secret": secret, "tolerance": tolerance})
        )
        return True

    async def create_vector_store_file(self, vector_store_id: str, *, file_id: str, **kwargs):
        self.workflow_calls.append(("create_vector_store_file", {"vector_store_id": vector_store_id, "file_id": file_id, **kwargs}))
        return VectorStoreFileResource(file_id=file_id, vector_store_id=vector_store_id)

    async def list_vector_store_files(self, vector_store_id: str, **kwargs):
        self.workflow_calls.append(("list_vector_store_files", {"vector_store_id": vector_store_id, **kwargs}))
        return VectorStoreFilesPage(items=[VectorStoreFileResource(file_id="file_1", vector_store_id=vector_store_id)])

    async def get_vector_store_file_content(self, vector_store_id: str, file_id: str, **kwargs):
        self.workflow_calls.append(("get_vector_store_file_content", {"vector_store_id": vector_store_id, "file_id": file_id, **kwargs}))
        return VectorStoreFileContentResult(file_id=file_id, vector_store_id=vector_store_id, chunks=[{"text": "hello"}])

    async def poll_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs):
        self.workflow_calls.append(("poll_vector_store_file", {"vector_store_id": vector_store_id, "file_id": file_id, **kwargs}))
        return VectorStoreFileResource(file_id=file_id, vector_store_id=vector_store_id, status="completed")

    async def create_vector_store_file_batch(self, vector_store_id: str, **kwargs):
        self.workflow_calls.append(("create_vector_store_file_batch", {"vector_store_id": vector_store_id, **kwargs}))
        return VectorStoreFileBatchResource(batch_id="vsfb_1", vector_store_id=vector_store_id, status="in_progress")

    async def list_vector_store_file_batch_files(self, vector_store_id: str, batch_id: str, **kwargs):
        self.workflow_calls.append(
            ("list_vector_store_file_batch_files", {"vector_store_id": vector_store_id, "batch_id": batch_id, **kwargs})
        )
        return VectorStoreFilesPage(items=[VectorStoreFileResource(file_id="file_2", vector_store_id=vector_store_id)])

    async def clarify_deep_research_task(self, prompt: str, **kwargs):
        self.workflow_calls.append(("clarify_deep_research_task", {"prompt": prompt, **kwargs}))
        return ok_result("What region and timeframe should I focus on?", model="gpt-4.1")

    async def rewrite_deep_research_prompt(self, prompt: str, **kwargs):
        self.workflow_calls.append(("rewrite_deep_research_prompt", {"prompt": prompt, **kwargs}))
        return ok_result("Rewritten deep research prompt", model="gpt-4.1")

    async def respond_with_web_search(self, prompt: str, **kwargs):
        self.workflow_calls.append(("respond_with_web_search", {"prompt": prompt, **kwargs}))
        return ok_result("web search done", model=self.model_name)

    async def respond_with_file_search(self, prompt: str, *, vector_store_ids: list[str] | tuple[str, ...], **kwargs):
        self.workflow_calls.append(
            ("respond_with_file_search", {"prompt": prompt, "vector_store_ids": list(vector_store_ids), **kwargs})
        )
        return ok_result("file search done", model=self.model_name)

    async def respond_with_code_interpreter(self, prompt: str, **kwargs):
        self.workflow_calls.append(("respond_with_code_interpreter", {"prompt": prompt, **kwargs}))
        return ok_result("code interpreter done", model=self.model_name)

    async def respond_with_remote_mcp(self, prompt: str, **kwargs):
        self.workflow_calls.append(("respond_with_remote_mcp", {"prompt": prompt, **kwargs}))
        return ok_result("remote mcp done", model=self.model_name)

    async def respond_with_connector(self, prompt: str, **kwargs):
        self.workflow_calls.append(("respond_with_connector", {"prompt": prompt, **kwargs}))
        return ok_result("connector done", model=self.model_name)

    async def start_deep_research(self, prompt: str, **kwargs):
        self.workflow_calls.append(("start_deep_research", {"prompt": prompt, **kwargs}))
        return ok_result("research queued", model=self.model_name)

    async def run_deep_research(self, prompt: str, **kwargs):
        self.workflow_calls.append(("run_deep_research", {"prompt": prompt, **kwargs}))
        queued = ok_result("research queued", model=self.model_name)
        return DeepResearchRunResult(
            prompt=prompt,
            effective_prompt=prompt,
            queued=queued,
            response_id="resp_1",
            background=BackgroundResponseResult(
                response_id="resp_1",
                lifecycle_status="completed",
                completion=ok_result("research final", model=self.model_name),
            ),
        )


@pytest.mark.asyncio
async def test_engine_retries_transient_complete_failure() -> None:
    provider = ScriptedProvider(complete_script=[error_result(500, "temporary"), ok_result("ok")])
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=2, backoff=0.0, max_backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert result.content == "ok"
    assert len(provider.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_falls_back_to_second_provider() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(router=StaticRouter([first, second]), retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert result.content == "from fallback"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 1


@pytest.mark.asyncio
async def test_engine_stream_falls_back_before_tokens_seen() -> None:
    first = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.ERROR, data={"status": 503, "error": "temporary"}),
        ]]
    )
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(
                type=StreamEventType.DONE,
                data=CompletionResult(content="hi", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2), model="gpt-5-mini", status=200),
            ),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]))

    events = [event async for event in engine.stream(_spec())]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert events[-1].data.content == "hi"
    assert len(first.stream_calls) == 1
    assert len(second.stream_calls) == 1


@pytest.mark.asyncio
async def test_engine_complete_honors_cancellation_before_attempt() -> None:
    provider = ScriptedProvider(complete_script=[ok_result("should not happen")])
    engine = ExecutionEngine(provider=provider)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        await engine.complete(_spec(), context=RequestContext(cancellation_token=token))

    assert provider.complete_calls == []


@pytest.mark.asyncio
async def test_engine_complete_standardizes_timeout_status() -> None:
    provider = _SlowProvider(delay=0.05)
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(_spec(), timeout=0.01)

    assert result.ok is False
    assert result.status == 408
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_engine_retries_standardized_retryable_statuses() -> None:
    provider = ScriptedProvider(complete_script=[error_result(425, "too early"), ok_result("ok")])
    engine = ExecutionEngine(provider=provider, retry=RetryConfig(attempts=2, backoff=0.0, max_backoff=0.0))

    result = await engine.complete(_spec())

    assert result.ok is True
    assert len(provider.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_complete_reuses_cached_idempotent_result() -> None:
    provider = ScriptedProvider(complete_script=[ok_result("once")])
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker)

    first = await engine.complete(_spec(), idempotency_key="idem-complete")
    second = await engine.complete(_spec(), idempotency_key="idem-complete")

    assert first.ok is True
    assert second.ok is True
    assert second.content == "once"
    assert len(provider.complete_calls) == 1
    assert tracker.completed_count == 1


@pytest.mark.asyncio
async def test_engine_stream_reuses_cached_idempotent_terminal_result() -> None:
    provider = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(type=StreamEventType.DONE, data=ok_result("hi")),
        ]]
    )
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker)

    first_events = [event async for event in engine.stream(_spec(), idempotency_key="idem-stream")]
    second_events = [event async for event in engine.stream(_spec(), idempotency_key="idem-stream")]

    assert [event.type for event in first_events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert [event.type for event in second_events] == [StreamEventType.DONE]
    assert second_events[0].data.content == "hi"
    assert len(provider.stream_calls) == 1
    assert tracker.completed_count == 1


@pytest.mark.asyncio
async def test_engine_stream_timeout_falls_back_before_tokens_seen() -> None:
    first = _SlowProvider(delay=0.05)
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(
                type=StreamEventType.DONE,
                data=CompletionResult(content="hi", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2), model="gpt-5-mini", status=200),
            ),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]))

    events = [event async for event in engine.stream(_spec(), timeout=0.01)]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert events[-1].data.content == "hi"


@pytest.mark.asyncio
async def test_engine_emits_router_selection_and_fallback_events() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        hooks=HookManager([hook]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )

    result = await engine.complete(_spec())

    assert result.ok is True
    event_names = [name for name, _ in hook.events]
    assert "router.selection" in event_names
    assert "router.fallback" in event_names
    selection_payload = next(payload for name, payload in hook.events if name == "router.selection")
    assert selection_payload["requested_model"] == "gpt-5-mini"
    assert selection_payload["selected_count"] == 2


@pytest.mark.asyncio
async def test_engine_request_end_payload_includes_usage_provider_and_model() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[ok_result("ok", model="gpt-5-mini")])
    engine = ExecutionEngine(provider=provider, hooks=HookManager([hook]))

    result = await engine.complete(_spec())

    assert result.ok is True
    request_end = next(payload for name, payload in hook.events if name == "request.end")
    assert request_end["provider"] == "openai"
    assert request_end["model"] == "gpt-5-mini"
    assert request_end["usage"]["total_tokens"] == result.usage.total_tokens


@pytest.mark.asyncio
async def test_engine_idempotent_request_end_payload_preserves_provider_model_and_usage() -> None:
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[ok_result("once", model="gpt-5-mini")])
    tracker = IdempotencyTracker()
    engine = ExecutionEngine(provider=provider, idempotency_tracker=tracker, hooks=HookManager([hook]))

    await engine.complete(_spec(), idempotency_key="idem-complete")
    second = await engine.complete(_spec(), idempotency_key="idem-complete")

    assert second.ok is True
    request_end_payloads = [payload for name, payload in hook.events if name == "request.end"]
    replay_end = request_end_payloads[-1]
    assert replay_end["provider"] == "openai"
    assert replay_end["model"] == "gpt-5-mini"
    assert replay_end["usage"]["total_tokens"] == second.usage.total_tokens


@pytest.mark.asyncio
async def test_engine_emits_request_lifecycle_hooks_and_diagnostics() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        hooks=HookManager([hook]),
        retry=RetryConfig(attempts=1, backoff=0.0),
    )

    result = await engine.complete(_spec())

    assert result.ok is True
    event_names = [name for name, _ in hook.events]
    assert event_names.count("request.pre_dispatch") == 2
    assert event_names.count("request.post_response") == 2
    assert "request.diagnostics" in event_names
    diagnostics = next(payload for name, payload in hook.events if name == "request.diagnostics")
    assert diagnostics["attempts"] == 2
    assert diagnostics["fallbacks"] == 1
    assert len(diagnostics["providers_selected"]) == 2
    assert len(diagnostics["providers_dispatched"]) == 2
    assert diagnostics["providers_tried"] == diagnostics["providers_dispatched"]
    assert diagnostics["final_status"] == 200
    assert diagnostics["cache_hit"] is False


@pytest.mark.asyncio
async def test_engine_emits_stream_lifecycle_hooks_and_diagnostics() -> None:
    hook = _CollectingHook()
    first = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.ERROR, data={"status": 503, "error": "temporary"}),
        ]]
    )
    second = ScriptedProvider(
        stream_script=[[
            StreamEvent(type=StreamEventType.TOKEN, data="hi"),
            StreamEvent(type=StreamEventType.DONE, data=ok_result("hi")),
        ]]
    )
    engine = ExecutionEngine(router=StaticRouter([first, second]), hooks=HookManager([hook]))

    events = [event async for event in engine.stream(_spec())]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    event_names = [name for name, _ in hook.events]
    assert event_names.count("stream.pre_dispatch") == 2
    assert "stream.diagnostics" in event_names
    diagnostics = next(payload for name, payload in hook.events if name == "stream.diagnostics")
    assert diagnostics["fallbacks"] == 1
    assert len(diagnostics["providers_selected"]) == 2
    assert len(diagnostics["providers_dispatched"]) == 2
    assert diagnostics["providers_tried"] == diagnostics["providers_dispatched"]
    assert diagnostics["token_seen"] is True
    assert diagnostics["final_status"] == 200


@pytest.mark.asyncio
async def test_engine_updates_registry_router_health_on_fallback() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")], model_name="gpt-5-mini")
    second = ScriptedProvider(complete_script=[ok_result("from fallback")], model_name="gemini-2.0-flash")
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: first,
            default_model="gpt-5-mini",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: second,
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
        )
    )
    router = RegistryRouter(registry=registry, unhealthy_after=1)
    engine = ExecutionEngine(router=router, retry=RetryConfig(attempts=1, backoff=0.0))

    result = await engine.complete(
        RequestSpec(
            provider="unknown",
            model=None,
            messages=[Message.user("hello")],
        )
    )

    assert result.ok is True
    assert router.get_provider_health("openai").degraded is True
    assert router.get_provider_health("google").successes == 1


@pytest.mark.asyncio
async def test_circuit_breaker_ignores_non_trip_statuses() -> None:
    provider = ScriptedProvider(complete_script=[error_result(400, "bad request"), error_result(400, "bad request"), ok_result("ok")])
    engine = ExecutionEngine(
        provider=provider,
        retry=RetryConfig(attempts=1, backoff=0.0),
        breaker_config=CircuitBreakerConfig(failure_threshold=1),
    )

    first = await engine.complete(_spec())
    second = await engine.complete(_spec())

    assert first.status == 400
    assert second.status == 400
    assert len(provider.complete_calls) == 2
    assert engine._get_breaker(engine._provider_id(provider)).get_state()["is_open"] is False


@pytest.mark.asyncio
async def test_engine_falls_back_when_primary_circuit_is_open() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")], model_name="gpt-5-mini")
    second = ScriptedProvider(
        complete_script=[ok_result("from fallback"), ok_result("from fallback")],
        model_name="gemini-2.0-flash",
    )
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        retry=RetryConfig(attempts=1, backoff=0.0),
        breaker_config=CircuitBreakerConfig(failure_threshold=1),
    )

    initial = await engine.complete(_spec())
    repeated = await engine.complete(_spec())

    assert initial.ok is True
    assert repeated.ok is True
    assert repeated.content == "from fallback"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 2


@pytest.mark.asyncio
async def test_engine_failover_policy_can_limit_provider_attempts() -> None:
    first = ScriptedProvider(complete_script=[error_result(503, "unavailable")])
    second = ScriptedProvider(complete_script=[ok_result("from fallback")])
    engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        retry=RetryConfig(attempts=1, backoff=0.0),
        failover_policy=FailoverPolicy(max_providers=1),
    )

    result = await engine.complete(_spec())

    assert result.ok is False
    assert result.status == 503
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 0


@pytest.mark.asyncio
async def test_engine_orchestrates_provider_workflow_methods() -> None:
    hook = _CollectingHook()
    provider = _WorkflowProvider()
    engine = ExecutionEngine(provider=provider, hooks=HookManager([hook]))

    retrieved = await engine.retrieve_background_response("resp_1")
    waited = await engine.wait_background_response("resp_1", poll_interval=0.0, timeout=5.0)
    cancelled = await engine.cancel_background_response("resp_2")
    conversation = await engine.create_conversation(items=[Message.user("hello")], metadata={"scope": "demo"})
    created_items = await engine.create_conversation_items("conv_1", items=[Message.user("hello")], include=["reasoning.encrypted_content"])
    listed_items = await engine.list_conversation_items("conv_1", after="item_0", limit=10, order="asc")
    retrieved_item = await engine.retrieve_conversation_item("conv_1", "item_2")
    deleted_item = await engine.delete_conversation_item("conv_1", "item_2")
    compacted = await engine.compact_response_context(messages=[Message.user("summarize")], model="gpt-5-mini")
    approved = await engine.submit_mcp_approval_response(
        previous_response_id="resp_prev",
        approval_request_id="apr_1",
        approve=True,
    )
    deleted_response = await engine.delete_response("resp_3")

    assert retrieved.lifecycle_status == "completed"
    assert waited.lifecycle_status == "completed"
    assert cancelled.lifecycle_status == "cancelled"
    assert conversation.conversation_id == "conv_1"
    assert created_items.items[0].content == "hello"
    assert listed_items.has_more is True
    assert retrieved_item.item_type == "mcp_approval_response"
    assert deleted_item.conversation_id == "conv_1"
    assert compacted.compaction_id == "cmp_1"
    assert approved.content == "approved"
    assert deleted_response.resource_id == "resp_3"

    event_names = [name for name, _ in hook.events]
    assert event_names.count("workflow.start") == 11
    assert event_names.count("workflow.end") == 11


@pytest.mark.asyncio
async def test_engine_orchestrates_background_stream_workflows() -> None:
    hook = _CollectingHook()
    provider = _WorkflowProvider()
    engine = ExecutionEngine(provider=provider, hooks=HookManager([hook]))

    events = [event async for event in engine.stream_background_response("resp_stream", starting_after=7)]

    assert [event.type for event in events] == [StreamEventType.TOKEN, StreamEventType.DONE]
    assert events[-1].data.content == "background done"
    assert provider.workflow_calls[-1] == (
        "stream_background_response",
        {"response_id": "resp_stream", "starting_after": 7},
    )
    event_names = [name for name, _ in hook.events]
    assert "workflow.start" in event_names
    assert "workflow.end" in event_names


@pytest.mark.asyncio
async def test_engine_orchestrates_extended_provider_api_workflows() -> None:
    provider = _WorkflowProvider()
    engine = ExecutionEngine(provider=provider)

    moderation = await engine.moderate("hello")
    image = await engine.generate_image("draw a cat")
    transcript = await engine.transcribe_audio("clip.wav")
    speech = await engine.synthesize_speech("hello", voice="alloy")
    vector_store = await engine.create_vector_store(name="Docs")
    search = await engine.search_vector_store("vs_1", query="cache invalidation")
    fine_tune = await engine.create_fine_tuning_job(model="gpt-4o-mini", training_file="file_1")
    events = await engine.list_fine_tuning_events("ftjob_1")

    assert moderation.flagged is True
    assert image.images[0].url == "https://example.com/image.png"
    assert transcript.text == "hello"
    assert speech.byte_length == 3
    assert vector_store.vector_store_id == "vs_1"
    assert search.results[0]["file_id"] == "file_1"
    assert fine_tune.job_id == "ftjob_1"
    assert events.events[0]["id"] == "ftevent_1"


@pytest.mark.asyncio
async def test_engine_orchestrates_realtime_webhook_vector_file_and_deep_research_workflows() -> None:
    provider = _WorkflowProvider()
    engine = ExecutionEngine(provider=provider)

    secret = await engine.create_realtime_client_secret(session={"type": "realtime"})
    transcription_secret = await engine.create_realtime_transcription_session(session={"type": "transcription"})
    connection = await engine.connect_realtime(model="gpt-realtime")
    transcription_connection = await engine.connect_realtime_transcription(model="gpt-realtime")
    uploaded_file = await engine.create_file(file="guide.pdf", purpose="assistants")
    retrieved_file = await engine.retrieve_file("file_1")
    listed_files = await engine.list_files()
    file_content = await engine.get_file_content("file_1")
    deleted_file = await engine.delete_file("file_1")
    call = await engine.create_realtime_call("offer-sdp", model="gpt-realtime")
    accepted = await engine.accept_realtime_call("rtc_1", type="realtime")
    verified = await engine.verify_webhook_signature("{}", {"webhook-signature": "sig"}, secret="whsec")
    webhook = await engine.unwrap_webhook("{}", {"webhook-signature": "sig"}, secret="whsec")
    vector_file = await engine.create_vector_store_file("vs_1", file_id="file_1")
    vector_files = await engine.list_vector_store_files("vs_1")
    vector_content = await engine.get_vector_store_file_content("vs_1", "file_1")
    polled_vector_file = await engine.poll_vector_store_file("vs_1", "file_1")
    vector_batch = await engine.create_vector_store_file_batch("vs_1", file_ids=["file_1", "file_2"])
    vector_batch_files = await engine.list_vector_store_file_batch_files("vs_1", "vsfb_1")
    clarification = await engine.clarify_deep_research_task("Research semaglutide")
    rewritten = await engine.rewrite_deep_research_prompt("Research semaglutide", clarifications=["Focus on Canada"])
    web_search = await engine.respond_with_web_search("Find latest docs")
    file_search = await engine.respond_with_file_search("Find tenant docs", vector_store_ids=["vs_1"])
    code_interpreter = await engine.respond_with_code_interpreter("Run analysis")
    remote_mcp = await engine.respond_with_remote_mcp("Inspect wiki", server_url="https://mcp.example.com")
    connector = await engine.respond_with_connector("Inspect gmail", connector_id="gmail")
    research = await engine.start_deep_research("Research semaglutide", web_search=True, rewrite_prompt=True)
    staged_research = await engine.run_deep_research("Research semaglutide", clarify_first=True, wait_for_completion=True)

    assert secret.value == "secret_1"
    assert transcription_secret.value == "tx_secret_1"
    assert connection.call_id == "rtc_live_1"
    assert transcription_connection.call_id == "rtc_tx_1"
    assert uploaded_file.file_id == "file_1"
    assert retrieved_file.filename == "guide.pdf"
    assert listed_files.items[0].purpose == "assistants"
    assert file_content.media_type == "application/pdf"
    assert deleted_file.deleted is True
    assert call.call_id == "rtc_1"
    assert accepted.action == "accept"
    assert verified is True
    assert webhook.event_type == "realtime.call.incoming"
    assert vector_file.file_id == "file_1"
    assert vector_files.items[0].vector_store_id == "vs_1"
    assert vector_content.chunks[0]["text"] == "hello"
    assert polled_vector_file.status == "completed"
    assert vector_batch.batch_id == "vsfb_1"
    assert vector_batch_files.items[0].file_id == "file_2"
    assert clarification.content.startswith("What region")
    assert rewritten.content == "Rewritten deep research prompt"
    assert web_search.content == "web search done"
    assert file_search.content == "file search done"
    assert code_interpreter.content == "code interpreter done"
    assert remote_mcp.content == "remote mcp done"
    assert connector.content == "connector done"
    assert research.content == "research queued"
    assert staged_research.background.completion.content == "research final"


@pytest.mark.asyncio
async def test_engine_passes_through_typed_retrieval_controls() -> None:
    provider = _WorkflowProvider()
    engine = ExecutionEngine(provider=provider)

    await engine.search_vector_store(
        "vs_1",
        query="cache invalidation",
        attribute_filter=ResponsesAttributeFilter.eq("scope", "tenant"),
        ranking_options=ResponsesFileSearchRankingOptions(score_threshold=0.2),
        max_num_results=7,
        rewrite_query=True,
    )
    await engine.respond_with_file_search(
        "Find tenant docs",
        vector_store_ids=["vs_1"],
        attribute_filter=ResponsesAttributeFilter.eq("scope", "tenant"),
        ranking_options=ResponsesFileSearchRankingOptions(score_threshold=0.15),
        max_num_results=4,
        include_search_results=True,
    )

    assert provider.workflow_calls[0] == (
        "search_vector_store",
        {
            "vector_store_id": "vs_1",
            "query": "cache invalidation",
            "attribute_filter": ResponsesAttributeFilter.eq("scope", "tenant"),
            "ranking_options": ResponsesFileSearchRankingOptions(score_threshold=0.2),
            "max_num_results": 7,
            "rewrite_query": True,
        },
    )
    assert provider.workflow_calls[1] == (
        "respond_with_file_search",
        {
            "prompt": "Find tenant docs",
            "vector_store_ids": ["vs_1"],
            "attribute_filter": ResponsesAttributeFilter.eq("scope", "tenant"),
            "ranking_options": ResponsesFileSearchRankingOptions(score_threshold=0.15),
            "max_num_results": 4,
            "include_search_results": True,
        },
    )
