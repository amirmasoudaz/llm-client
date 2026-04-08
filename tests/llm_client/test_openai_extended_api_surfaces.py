from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.openai import OpenAIProvider
from llm_client.tools import (
    ResponsesAttributeFilter,
    ResponsesFileSearchHybridWeights,
    ResponsesFileSearchRankingOptions,
    ResponsesMCPTool,
)
from tests.llm_client.fakes import FakeModel


class _LimitContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _NoopLimiter:
    def limit(self, **kwargs):
        _ = kwargs
        return _LimitContext()


class _SinglePage:
    def __init__(self, page: object) -> None:
        self._page = page

    async def _get_page(self):
        return self._page


class _FakeRealtimeConnection:
    def __init__(self, *, recv_events: list[object] | None = None, recv_bytes_values: list[bytes] | None = None) -> None:
        self.sent: list[object] = []
        self.closed = False
        self._recv_events = list(recv_events or [SimpleNamespace(to_dict=lambda: {"type": "session.updated"})])
        self._recv_bytes_values = list(recv_bytes_values or [b"audio"])

    async def send(self, event) -> None:
        self.sent.append(event)

    async def recv(self):
        if self._recv_events:
            return self._recv_events.pop(0)
        return SimpleNamespace(to_dict=lambda: {"type": "session.updated"})

    async def recv_bytes(self) -> bytes:
        if self._recv_bytes_values:
            return self._recv_bytes_values.pop(0)
        return b"audio"

    async def close(self) -> None:
        self.closed = True


class _FakeRealtimeManager:
    def __init__(self, connection: _FakeRealtimeConnection) -> None:
        self.connection = connection
        self.exited = False

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        self.exited = True
        await self.connection.close()


def _openai_provider(model_name: str) -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._model = FakeModel(key=model_name, model_name=model_name)
    provider.limiter = _NoopLimiter()
    return provider


@pytest.mark.asyncio
async def test_openai_moderate_parses_results() -> None:
    provider = _openai_provider("omni-moderation-latest")

    async def _moderate_create(**kwargs):
        assert kwargs["input"] == "hello"
        return SimpleNamespace(
            model="omni-moderation-latest",
            results=[
                SimpleNamespace(
                    flagged=True,
                    categories={"violence": True},
                    category_scores={"violence": 0.93},
                    category_applied_input_types={"violence": ["text"]},
                )
            ],
        )

    provider.client = SimpleNamespace(moderations=SimpleNamespace(create=_moderate_create))

    result = await provider.moderate("hello")

    assert result.flagged is True
    assert result.model == "omni-moderation-latest"
    assert result.results[0]["categories"]["violence"] is True


@pytest.mark.asyncio
async def test_openai_image_generation_and_edit_surfaces() -> None:
    provider = _openai_provider("gpt-image-1")

    async def _generate(**kwargs):
        assert kwargs["prompt"] == "draw a lighthouse"
        return SimpleNamespace(
            created=123,
            model="gpt-image-1",
            data=[SimpleNamespace(url="https://example.com/a.png", revised_prompt="more cinematic")],
        )

    async def _edit(**kwargs):
        assert kwargs["prompt"] == "replace the sky"
        return SimpleNamespace(
            created=124,
            model="gpt-image-1",
            data=[SimpleNamespace(b64_json="YWJj", revised_prompt="masked edit")],
        )

    provider.client = SimpleNamespace(images=SimpleNamespace(generate=_generate, edit=_edit))

    generated = await provider.generate_image("draw a lighthouse")
    edited = await provider.edit_image("image.png", "replace the sky")

    assert generated.images[0].url == "https://example.com/a.png"
    assert generated.images[0].revised_prompt == "more cinematic"
    assert edited.images[0].b64_json == "YWJj"


@pytest.mark.asyncio
async def test_openai_audio_transcription_translation_and_speech_surfaces() -> None:
    provider = _openai_provider("gpt-4o-transcribe")

    async def _transcribe(**kwargs):
        assert kwargs["file"] == "clip.wav"
        return SimpleNamespace(
            text="hello world",
            language="en",
            duration=1.5,
            words=[{"word": "hello", "start": 0.0, "end": 0.5}],
        )

    async def _translate(**kwargs):
        assert kwargs["file"] == "clip-es.wav"
        return SimpleNamespace(text="hello translated", language="en")

    async def _speech(**kwargs):
        assert kwargs["input"] == "speak now"
        assert kwargs["voice"] == "alloy"
        return SimpleNamespace(content=b"abc")

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=_transcribe),
            translations=SimpleNamespace(create=_translate),
            speech=SimpleNamespace(create=_speech),
        )
    )

    transcript = await provider.transcribe_audio("clip.wav")
    translation = await provider.translate_audio("clip-es.wav")
    speech = await provider.synthesize_speech("speak now", voice="alloy", model="tts-1")

    assert transcript.text == "hello world"
    assert transcript.language == "en"
    assert translation.text == "hello translated"
    assert speech.audio == b"abc"
    assert speech.format == "mp3"


@pytest.mark.asyncio
async def test_openai_vector_store_and_fine_tuning_surfaces() -> None:
    provider = _openai_provider("gpt-4o-mini")

    async def _create_vector_store(**kwargs):
        assert kwargs["name"] == "Docs"
        return SimpleNamespace(id="vs_1", name="Docs", status="completed", file_counts={"completed": 1}, usage_bytes=128)

    async def _search_vector_store(vector_store_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        return _SinglePage(
            {
                "data": [
                    {"file_id": "file_1", "filename": "guide.md", "score": 0.9, "content": [{"text": "hello"}]}
                ]
            }
        )

    async def _create_job(**kwargs):
        assert kwargs["training_file"] == "file_train"
        return SimpleNamespace(id="ftjob_1", status="queued", model="gpt-4o-mini", training_file="file_train")

    async def _list_events(job_id: str, **kwargs):
        assert job_id == "ftjob_1"
        return _SinglePage({"data": [{"id": "ftevent_1", "message": "queued"}], "has_more": False})

    provider.client = SimpleNamespace(
        vector_stores=SimpleNamespace(
            create=_create_vector_store,
            search=_search_vector_store,
        ),
        fine_tuning=SimpleNamespace(
            jobs=SimpleNamespace(
                create=_create_job,
                list_events=_list_events,
            )
        ),
    )

    vector_store = await provider.create_vector_store(name="Docs")
    search = await provider.search_vector_store("vs_1", query="hello")
    job = await provider.create_fine_tuning_job(model="gpt-4o-mini", training_file="file_train")
    events = await provider.list_fine_tuning_events("ftjob_1")

    assert vector_store.vector_store_id == "vs_1"
    assert search.results[0]["file_id"] == "file_1"
    assert job.job_id == "ftjob_1"
    assert events.events[0]["id"] == "ftevent_1"


@pytest.mark.asyncio
async def test_openai_vector_store_search_supports_typed_retrieval_controls() -> None:
    provider = _openai_provider("gpt-4o-mini")

    async def _search_vector_store(vector_store_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        assert kwargs["query"] == "hello"
        assert kwargs["filters"] == {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "scope", "value": "tenant"},
                {"type": "gte", "key": "priority", "value": 0.8},
            ],
        }
        assert kwargs["ranking_options"] == {
            "ranker": "default-2024-11-15",
            "score_threshold": 0.3,
            "hybrid_search": {"embedding_weight": 0.6, "text_weight": 0.4},
        }
        assert kwargs["max_num_results"] == 8
        assert kwargs["rewrite_query"] is True
        return _SinglePage({"data": [{"file_id": "file_1", "score": 0.91, "filename": "guide.md"}]})

    provider.client = SimpleNamespace(
        vector_stores=SimpleNamespace(
            search=_search_vector_store,
        )
    )

    search = await provider.search_vector_store(
        "vs_1",
        query="hello",
        attribute_filter=ResponsesAttributeFilter.and_(
            ResponsesAttributeFilter.eq("scope", "tenant"),
            ResponsesAttributeFilter.gte("priority", 0.8),
        ),
        ranking_options=ResponsesFileSearchRankingOptions(
            ranker="default-2024-11-15",
            score_threshold=0.3,
            hybrid_search=ResponsesFileSearchHybridWeights(embedding_weight=0.6, text_weight=0.4),
        ),
        max_num_results=8,
        rewrite_query=True,
    )

    assert search.vector_store_id == "vs_1"
    assert search.results[0]["file_id"] == "file_1"


@pytest.mark.asyncio
async def test_openai_generic_file_surfaces() -> None:
    provider = _openai_provider("gpt-4o-mini")

    async def _create(**kwargs):
        assert kwargs["file"] == "guide.pdf"
        assert kwargs["purpose"] == "assistants"
        return SimpleNamespace(id="file_1", filename="guide.pdf", purpose="assistants", bytes=128, status="processed")

    async def _retrieve(file_id: str, **kwargs):
        assert file_id == "file_1"
        return SimpleNamespace(id=file_id, filename="guide.pdf", purpose="assistants", bytes=128, status="processed")

    async def _list(**kwargs):
        assert kwargs["purpose"] == "assistants"
        return _SinglePage(
            {
                "data": [
                    {"id": "file_1", "filename": "guide.pdf", "purpose": "assistants", "bytes": 128, "status": "processed"}
                ],
                "first_id": "file_1",
                "last_id": "file_1",
                "has_more": False,
            }
        )

    async def _delete(file_id: str, **kwargs):
        assert file_id == "file_1"
        return SimpleNamespace(id=file_id, deleted=True)

    async def _content(file_id: str, **kwargs):
        assert file_id == "file_1"
        return SimpleNamespace(content=b"abc", headers={"content-type": "application/pdf"})

    provider.client = SimpleNamespace(
        files=SimpleNamespace(
            create=_create,
            retrieve=_retrieve,
            list=_list,
            delete=_delete,
            content=_content,
        )
    )

    created = await provider.create_file(file="guide.pdf", purpose="assistants")
    retrieved = await provider.retrieve_file("file_1")
    listed = await provider.list_files(purpose="assistants")
    deleted = await provider.delete_file("file_1")
    content = await provider.get_file_content("file_1")

    assert created.file_id == "file_1"
    assert retrieved.filename == "guide.pdf"
    assert listed.items[0].purpose == "assistants"
    assert deleted.deleted is True
    assert content.content == b"abc"
    assert content.media_type == "application/pdf"


@pytest.mark.asyncio
async def test_openai_realtime_and_webhook_surfaces() -> None:
    provider = _openai_provider("gpt-realtime")
    realtime_connection = _FakeRealtimeConnection(
        recv_events=[
            SimpleNamespace(to_dict=lambda: {"type": "session.updated"}),
            SimpleNamespace(
                to_dict=lambda: {
                    "type": "conversation.item.added",
                    "event_id": "evt_1",
                    "item": {"id": "item_1", "type": "message", "status": "in_progress"},
                    "previous_item_id": "item_0",
                }
            ),
            SimpleNamespace(
                to_dict=lambda: {
                    "type": "response.output_text.delta",
                    "event_id": "evt_2",
                    "response_id": "resp_1",
                    "item_id": "item_1",
                    "delta": "hello",
                    "content_index": 0,
                }
            ),
            SimpleNamespace(
                to_dict=lambda: {
                    "type": "response.done",
                    "event_id": "evt_3",
                    "response_id": "resp_1",
                    "status": "completed",
                }
            ),
        ]
    )
    realtime_manager = _FakeRealtimeManager(realtime_connection)
    transcription_connection = _FakeRealtimeConnection()
    transcription_manager = _FakeRealtimeManager(transcription_connection)

    async def _create_secret(**kwargs):
        if kwargs["session"] == {"type": "realtime"}:
            return SimpleNamespace(
                client_secret=SimpleNamespace(value="rt_secret", expires_at=1234),
                session={"type": "realtime"},
            )
        assert kwargs["session"] == {
            "type": "transcription",
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-realtime"},
                }
            },
        }
        return SimpleNamespace(
            client_secret=SimpleNamespace(value="rt_tx_secret", expires_at=5678),
            session=kwargs["session"],
        )

    async def _create_call(*, sdp: str, **kwargs):
        assert sdp == "offer-sdp"
        assert kwargs["model"] == "gpt-realtime"
        return SimpleNamespace(
            response=SimpleNamespace(headers={"Location": "/v1/realtime/calls/rtc_1"}, status_code=201),
            text="answer-sdp",
        )

    async def _accept(call_id: str, **kwargs):
        assert call_id == "rtc_1"
        assert kwargs["session"] == {"type": "realtime"}
        return None

    connect_calls: list[dict[str, object]] = []

    connect_calls: list[dict[str, object]] = []

    def _connect(**kwargs):
        assert kwargs["model"] == "gpt-realtime"
        connect_calls.append(dict(kwargs))
        if len(connect_calls) == 2:
            return transcription_manager
        return realtime_manager

    webhook_calls: list[tuple[str, object]] = []

    def _unwrap(payload, headers, *, secret=None):
        webhook_calls.append(("unwrap", secret))
        assert payload == "{}"
        assert headers["webhook-signature"] == "sig"
        return {"id": "evt_1", "type": "response.completed", "data": {"response_id": "resp_1"}}

    def _verify(payload, headers, *, secret=None, tolerance=300):
        webhook_calls.append(("verify", {"secret": secret, "tolerance": tolerance}))
        assert payload == "{}"
        assert headers["webhook-signature"] == "sig"
        return None

    provider.client = SimpleNamespace(
        realtime=SimpleNamespace(
            client_secrets=SimpleNamespace(create=_create_secret),
            calls=SimpleNamespace(create=_create_call, accept=_accept),
            connect=_connect,
        ),
        webhooks=SimpleNamespace(unwrap=_unwrap, verify_signature=_verify),
    )

    secret = await provider.create_realtime_client_secret(session={"type": "realtime"})
    transcription_secret = await provider.create_realtime_transcription_session(session={"type": "transcription"})
    connection = await provider.connect_realtime(model="gpt-realtime")
    transcription_stream = await provider.connect_realtime_transcription(model="gpt-realtime")
    call = await provider.create_realtime_call("offer-sdp", model="gpt-realtime")
    accepted = await provider.accept_realtime_call("rtc_1", session={"type": "realtime"})
    verified = await provider.verify_webhook_signature("{}", {"webhook-signature": "sig"}, secret="whsec", tolerance=42)
    event = await provider.unwrap_webhook("{}", {"webhook-signature": "sig"}, secret="whsec")
    await connection.update_session({"modalities": ["text"]}, event_id="evt_session")
    await connection.create_response({"modalities": ["text"]}, event_id="evt_response_create")
    await connection.create_conversation_item(
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        event_id="evt_create",
    )
    await connection.retrieve_conversation_item("item_1", event_id="evt_retrieve")
    await connection.delete_conversation_item("item_2", event_id="evt_delete")
    await connection.truncate_conversation_item("item_3", audio_end_ms=1200, event_id="evt_truncate")
    await connection.append_input_audio(b"abc", event_id="evt_append")
    await connection.cancel_response(response_id="resp_1", event_id="evt_cancel")
    await connection.commit_input_audio(event_id="evt_commit")
    await connection.clear_input_audio(event_id="evt_clear_input")
    await connection.clear_output_audio(event_id="evt_clear_output")
    received = await connection.recv()
    received_event = await connection.recv_event()
    received_delta = await connection.recv_until_type("response.output_text.delta", timeout=1.0)
    received_done = await connection.recv_until_type("response.done", timeout=1.0)
    received_bytes = await connection.recv_bytes()
    await connection.close()
    await transcription_stream.close()

    assert secret.value == "rt_secret"
    assert transcription_secret.value == "rt_tx_secret"
    assert connection.model == "gpt-realtime"
    assert transcription_stream.model == "gpt-realtime"
    assert call.call_id == "rtc_1"
    assert call.sdp == "answer-sdp"
    assert call.status == 201
    assert accepted.call_id == "rtc_1"
    assert accepted.action == "accept"
    assert verified is True
    assert event.event_id == "evt_1"
    assert event.event_type == "response.completed"
    assert event.data == {"response_id": "resp_1"}
    assert realtime_connection.sent == [
        {"type": "session.update", "session": {"modalities": ["text"]}, "event_id": "evt_session"},
        {"type": "response.create", "response": {"modalities": ["text"]}, "event_id": "evt_response_create"},
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            "event_id": "evt_create",
        },
        {"type": "conversation.item.retrieve", "item_id": "item_1", "event_id": "evt_retrieve"},
        {"type": "conversation.item.delete", "item_id": "item_2", "event_id": "evt_delete"},
        {
            "type": "conversation.item.truncate",
            "item_id": "item_3",
            "content_index": 0,
            "audio_end_ms": 1200,
            "event_id": "evt_truncate",
        },
        {"type": "input_audio_buffer.append", "audio": "YWJj", "event_id": "evt_append"},
        {"type": "response.cancel", "response_id": "resp_1", "event_id": "evt_cancel"},
        {"type": "input_audio_buffer.commit", "event_id": "evt_commit"},
        {"type": "input_audio_buffer.clear", "event_id": "evt_clear_input"},
        {"type": "output_audio_buffer.clear", "event_id": "evt_clear_output"},
    ]
    assert received == {"type": "session.updated"}
    assert received_event.event_type == "conversation.item.added"
    assert received_event.event_id == "evt_1"
    assert received_event.item == {"id": "item_1", "type": "message", "status": "in_progress"}
    assert received_event.previous_item_id == "item_0"
    assert received_delta.event_type == "response.output_text.delta"
    assert received_delta.response_id == "resp_1"
    assert received_delta.item_id == "item_1"
    assert received_delta.delta == "hello"
    assert received_delta.details["content_index"] == 0
    assert received_done.event_type == "response.done"
    assert received_done.event_id == "evt_3"
    assert received_done.response_id == "resp_1"
    assert received_done.status == "completed"
    assert received_bytes == b"audio"
    assert realtime_manager.exited is True
    assert transcription_manager.exited is True
    assert webhook_calls == [
        ("verify", {"secret": "whsec", "tolerance": 42}),
        ("unwrap", "whsec"),
    ]


@pytest.mark.asyncio
async def test_openai_realtime_transcription_connection_normalizes_transcription_models() -> None:
    provider = _openai_provider("gpt-4o-mini-transcribe")
    transcription_connection = _FakeRealtimeConnection()
    transcription_manager = _FakeRealtimeManager(transcription_connection)
    captured: dict[str, object] = {}

    def _connect(**kwargs):
        captured.update(kwargs)
        return transcription_manager

    provider.client = SimpleNamespace(
        realtime=SimpleNamespace(connect=_connect),
    )

    connection = await provider.connect_realtime_transcription()
    await connection.close()

    assert captured["model"] == "gpt-realtime"
    assert connection.model == "gpt-realtime"


@pytest.mark.asyncio
async def test_openai_vector_store_file_surfaces() -> None:
    provider = _openai_provider("gpt-4o-mini")

    async def _create(vector_store_id: str, *, file_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        assert file_id == "file_1"
        assert kwargs["attributes"] == {"scope": "docs"}
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, status="completed", usage_bytes=256)

    async def _upload(*, vector_store_id: str, file, **kwargs):
        assert vector_store_id == "vs_1"
        assert file == "guide.md"
        assert kwargs["attributes"] == {"kind": "upload"}
        return SimpleNamespace(id="file_2", vector_store_id=vector_store_id, status="in_progress")

    async def _list(vector_store_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        assert kwargs["limit"] == 10
        return _SinglePage(
            {
                "data": [
                    {"id": "file_1", "vector_store_id": "vs_1", "status": "completed"},
                    {"id": "file_2", "vector_store_id": "vs_1", "status": "in_progress"},
                ],
                "first_id": "file_1",
                "last_id": "file_2",
                "has_more": True,
            }
        )

    async def _retrieve(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_1"
        assert vector_store_id == "vs_1"
        assert kwargs["include"] == ["attributes"]
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, status="completed", attributes={"scope": "docs"})

    async def _update(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_1"
        assert vector_store_id == "vs_1"
        assert kwargs["attributes"] == {"scope": "updated"}
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, status="completed", attributes={"scope": "updated"})

    async def _delete(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_2"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=file_id, deleted=True)

    async def _content(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_1"
        assert vector_store_id == "vs_1"
        return _SinglePage({"data": [{"text": "chunk 1"}, {"text": "chunk 2"}]})

    async def _poll(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_1"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, status="completed")

    async def _create_and_poll(file_id: str, *, vector_store_id: str, **kwargs):
        assert file_id == "file_3"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=file_id, vector_store_id=vector_store_id, status="completed")

    async def _upload_and_poll(*, vector_store_id: str, file, **kwargs):
        assert vector_store_id == "vs_1"
        assert file == "ready.md"
        return SimpleNamespace(id="file_4", vector_store_id=vector_store_id, status="completed")

    async def _batch_create(vector_store_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        assert kwargs["file_ids"] == ["file_1", "file_2"]
        return SimpleNamespace(id="vsfb_1", vector_store_id=vector_store_id, status="in_progress", file_counts={"in_progress": 2})

    async def _batch_retrieve(batch_id: str, *, vector_store_id: str, **kwargs):
        assert batch_id == "vsfb_1"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=batch_id, vector_store_id=vector_store_id, status="completed", file_counts={"completed": 2})

    async def _batch_cancel(batch_id: str, *, vector_store_id: str, **kwargs):
        assert batch_id == "vsfb_2"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=batch_id, vector_store_id=vector_store_id, status="cancelled", file_counts={"cancelled": 1})

    async def _batch_poll(batch_id: str, *, vector_store_id: str, **kwargs):
        assert batch_id == "vsfb_1"
        assert vector_store_id == "vs_1"
        return SimpleNamespace(id=batch_id, vector_store_id=vector_store_id, status="completed", file_counts={"completed": 2})

    async def _batch_list_files(batch_id: str, *, vector_store_id: str, **kwargs):
        assert batch_id == "vsfb_1"
        assert vector_store_id == "vs_1"
        return _SinglePage({"data": [{"id": "file_10", "vector_store_id": "vs_1", "status": "completed"}]})

    async def _batch_create_and_poll(vector_store_id: str, **kwargs):
        assert vector_store_id == "vs_1"
        assert kwargs["file_ids"] == ["file_5"]
        return SimpleNamespace(id="vsfb_3", vector_store_id=vector_store_id, status="completed", file_counts={"completed": 1})

    async def _batch_upload_and_poll(vector_store_id: str, *, files, **kwargs):
        assert vector_store_id == "vs_1"
        assert list(files) == ["a.txt", "b.txt"]
        return SimpleNamespace(id="vsfb_4", vector_store_id=vector_store_id, status="completed", file_counts={"completed": 2})

    provider.client = SimpleNamespace(
        vector_stores=SimpleNamespace(
            files=SimpleNamespace(
                create=_create,
                upload=_upload,
                list=_list,
                retrieve=_retrieve,
                update=_update,
                delete=_delete,
                content=_content,
                poll=_poll,
                create_and_poll=_create_and_poll,
                upload_and_poll=_upload_and_poll,
            )
            ,
            file_batches=SimpleNamespace(
                create=_batch_create,
                retrieve=_batch_retrieve,
                cancel=_batch_cancel,
                poll=_batch_poll,
                list_files=_batch_list_files,
                create_and_poll=_batch_create_and_poll,
                upload_and_poll=_batch_upload_and_poll,
            ),
        )
    )

    created = await provider.create_vector_store_file("vs_1", file_id="file_1", attributes={"scope": "docs"})
    uploaded = await provider.upload_vector_store_file("vs_1", file="guide.md", attributes={"kind": "upload"})
    listed = await provider.list_vector_store_files("vs_1", limit=10)
    retrieved = await provider.retrieve_vector_store_file("vs_1", "file_1", include=["attributes"])
    updated = await provider.update_vector_store_file("vs_1", "file_1", attributes={"scope": "updated"})
    deleted = await provider.delete_vector_store_file("vs_1", "file_2")
    content = await provider.get_vector_store_file_content("vs_1", "file_1")
    polled = await provider.poll_vector_store_file("vs_1", "file_1")
    created_polled = await provider.create_vector_store_file_and_poll("vs_1", file_id="file_3")
    uploaded_polled = await provider.upload_vector_store_file_and_poll("vs_1", file="ready.md")
    batch = await provider.create_vector_store_file_batch("vs_1", file_ids=["file_1", "file_2"])
    retrieved_batch = await provider.retrieve_vector_store_file_batch("vs_1", "vsfb_1")
    cancelled_batch = await provider.cancel_vector_store_file_batch("vs_1", "vsfb_2")
    polled_batch = await provider.poll_vector_store_file_batch("vs_1", "vsfb_1")
    batch_files = await provider.list_vector_store_file_batch_files("vs_1", "vsfb_1")
    created_batch_polled = await provider.create_vector_store_file_batch_and_poll("vs_1", file_ids=["file_5"])
    uploaded_batch_polled = await provider.upload_vector_store_file_batch_and_poll("vs_1", files=["a.txt", "b.txt"])

    assert created.file_id == "file_1"
    assert created.usage_bytes == 256
    assert uploaded.file_id == "file_2"
    assert listed.first_id == "file_1"
    assert listed.has_more is True
    assert [item.file_id for item in listed.items] == ["file_1", "file_2"]
    assert retrieved.attributes == {"scope": "docs"}
    assert updated.attributes == {"scope": "updated"}
    assert deleted.deleted is True
    assert [chunk["text"] for chunk in content.chunks] == ["chunk 1", "chunk 2"]
    assert polled.status == "completed"
    assert created_polled.file_id == "file_3"
    assert uploaded_polled.file_id == "file_4"
    assert batch.batch_id == "vsfb_1"
    assert retrieved_batch.status == "completed"
    assert cancelled_batch.status == "cancelled"
    assert polled_batch.file_counts == {"completed": 2}
    assert batch_files.items[0].file_id == "file_10"
    assert created_batch_polled.batch_id == "vsfb_3"
    assert uploaded_batch_polled.batch_id == "vsfb_4"


@pytest.mark.asyncio
async def test_openai_start_deep_research_requires_data_source_and_builds_tools() -> None:
    provider = _openai_provider("o3-deep-research")
    provider.use_responses_api = True
    captured: list[dict[str, object]] = []

    async def _complete(messages, **kwargs):
        captured.append({"messages": messages, **kwargs})
        return SimpleNamespace(ok=True, content="queued")

    provider.complete = _complete  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        await provider.start_deep_research("Research semaglutide")

    result = await provider.start_deep_research(
        "Research semaglutide",
        model="o3-deep-research",
        web_search=True,
        vector_store_ids=["vs_1"],
        mcp_tools=[{"type": "mcp", "server_label": "docs"}],
        include_code_interpreter=True,
    )

    assert getattr(result, "ok") is True
    assert len(captured) == 1
    call = captured[0]
    rendered_tools = [
        tool.to_dict() if hasattr(tool, "to_dict") else tool
        for tool in call["tools"]
    ]
    assert call["messages"] == "Research semaglutide"
    assert call["model"] == "o3-deep-research"
    assert call["background"] is True
    assert {"type": "web_search_preview"} in rendered_tools
    assert {"type": "file_search", "vector_store_ids": ["vs_1"]} in rendered_tools
    assert {"type": "mcp", "server_label": "docs", "require_approval": "never"} in rendered_tools
    assert {"type": "code_interpreter", "container": {"type": "auto"}} in rendered_tools


@pytest.mark.asyncio
async def test_openai_deep_research_clarify_and_rewrite_helpers() -> None:
    provider = _openai_provider("o3-deep-research")
    captured: list[dict[str, object]] = []

    async def _complete(messages, **kwargs):
        captured.append({"messages": messages, **kwargs})
        if kwargs["model"] == "gpt-4.1":
            return SimpleNamespace(ok=True, content="Rewritten prompt")
        return SimpleNamespace(ok=True, content="queued")

    provider.complete = _complete  # type: ignore[method-assign]
    provider.use_responses_api = True

    clarification = await provider.clarify_deep_research_task("Research surfboards")
    rewritten = await provider.rewrite_deep_research_prompt(
        "Research surfboards",
        clarifications=["Budget under $900", "Target beginner/intermediate rider"],
    )
    result = await provider.start_deep_research(
        "Research surfboards",
        web_search=True,
        rewrite_prompt=True,
        clarifications=["Budget under $900"],
        mcp_tools=[{"type": "mcp", "server_label": "docs"}],
    )

    assert clarification.content == "Rewritten prompt"
    assert rewritten.content == "Rewritten prompt"
    assert result.content == "queued"
    assert "Research surfboards" in captured[1]["messages"]
    assert "Budget under $900" in captured[1]["messages"]
    assert "Research surfboards" in captured[2]["messages"]
    assert captured[3]["messages"] == "Rewritten prompt"
    rendered_tools = [
        tool.to_dict() if hasattr(tool, "to_dict") else tool
        for tool in captured[3]["tools"]
    ]
    assert {"type": "mcp", "server_label": "docs", "require_approval": "never"} in rendered_tools


@pytest.mark.asyncio
async def test_openai_deep_research_rejects_mcp_approval_modes_other_than_never() -> None:
    provider = _openai_provider("o3-deep-research")
    provider.use_responses_api = True

    async def _complete(messages, **kwargs):
        return SimpleNamespace(ok=True, content="queued")

    provider.complete = _complete  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        await provider.start_deep_research(
            "Research semaglutide",
            web_search=True,
            mcp_tools=[{"type": "mcp", "server_label": "docs", "require_approval": "always"}],
        )


@pytest.mark.asyncio
async def test_openai_hosted_tool_workflow_helpers_build_typed_tools() -> None:
    provider = _openai_provider("gpt-5-mini")
    provider.use_responses_api = True
    captured: list[dict[str, object]] = []

    async def _complete(messages, **kwargs):
        captured.append({"messages": messages, **kwargs})
        return SimpleNamespace(ok=True, content="done")

    provider.complete = _complete  # type: ignore[method-assign]

    remote_mcp_tool = ResponsesMCPTool.remote_server(
        "https://mcp.example.com",
        server_label="Research Wiki",
        require_approval="always",
    )

    await provider.respond_with_web_search("Find latest docs", tool_config={"search_context_size": "low"})
    await provider.respond_with_file_search("Search my files", vector_store_ids=["vs_1"], tool_config={"max_num_results": 3})
    await provider.respond_with_code_interpreter("Run an analysis")
    await provider.respond_with_remote_mcp("Inspect wiki", tool=remote_mcp_tool)
    await provider.respond_with_connector(
        "Inspect gmail",
        connector_id="connector_gmail",
        server_label="Gmail",
        authorization="Bearer oauth-token",
    )

    rendered_calls = [
        [tool.to_dict() if hasattr(tool, "to_dict") else tool for tool in call["tools"]]
        for call in captured
    ]

    assert rendered_calls[0] == [{"type": "web_search_preview", "search_context_size": "low"}]
    assert rendered_calls[1] == [{"type": "file_search", "vector_store_ids": ["vs_1"], "max_num_results": 3}]
    assert rendered_calls[2] == [{"type": "code_interpreter", "container": {"type": "auto"}}]
    assert rendered_calls[3] == [
        {
            "type": "mcp",
            "server_label": "Research Wiki",
            "server_url": "https://mcp.example.com",
            "require_approval": "always",
        }
    ]
    assert rendered_calls[4] == [
        {
            "type": "mcp",
            "connector_id": "connector_gmail",
            "server_label": "Gmail",
            "authorization": "Bearer oauth-token",
        }
    ]


@pytest.mark.asyncio
async def test_openai_file_search_helper_supports_typed_retrieval_controls_and_include() -> None:
    provider = _openai_provider("gpt-5-mini")
    provider.use_responses_api = True
    captured: list[dict[str, object]] = []

    async def _complete(messages, **kwargs):
        captured.append({"messages": messages, **kwargs})
        return SimpleNamespace(ok=True, content="done")

    provider.complete = _complete  # type: ignore[method-assign]

    await provider.respond_with_file_search(
        "Search my files",
        vector_store_ids=["vs_1"],
        attribute_filter=ResponsesAttributeFilter.eq("scope", "tenant"),
        ranking_options=ResponsesFileSearchRankingOptions(score_threshold=0.15),
        max_num_results=6,
        include_search_results=True,
        include=["output_text"],
    )

    rendered_tools = [tool.to_dict() if hasattr(tool, "to_dict") else tool for tool in captured[0]["tools"]]

    assert rendered_tools == [
        {
            "type": "file_search",
            "vector_store_ids": ["vs_1"],
            "filters": {"type": "eq", "key": "scope", "value": "tenant"},
            "ranking_options": {"score_threshold": 0.15},
            "max_num_results": 6,
        }
    ]
    assert captured[0]["include"] == ["output_text", "file_search_call.results"]


@pytest.mark.asyncio
async def test_openai_file_search_helper_rejects_mixed_explicit_tool_and_helper_controls() -> None:
    provider = _openai_provider("gpt-5-mini")
    provider.use_responses_api = True

    with pytest.raises(ValueError, match="Provide file-search tuning controls on the explicit `tool` object"):
        await provider.respond_with_file_search(
            "Search my files",
            vector_store_ids=["vs_1"],
            tool=ResponsesMCPTool.remote_server("https://mcp.example.com", server_label="Wiki"),
            attribute_filter=ResponsesAttributeFilter.eq("scope", "tenant"),
        )


@pytest.mark.asyncio
async def test_openai_run_deep_research_orchestrates_rewrite_and_wait() -> None:
    provider = _openai_provider("o3-deep-research")
    provider.use_responses_api = True

    async def _clarify(prompt, **kwargs):
        assert prompt == "Research semaglutide"
        return SimpleNamespace(ok=True, content="What region and timeframe?")

    async def _rewrite(prompt, **kwargs):
        assert kwargs["clarifications"] == ["Canada", "2025"]
        return SimpleNamespace(ok=True, content="Research semaglutide in Canada during 2025")

    async def _start(prompt, **kwargs):
        assert prompt == "Research semaglutide in Canada during 2025"
        assert kwargs["rewrite_prompt"] is False
        return SimpleNamespace(ok=True, content="queued", raw_response=SimpleNamespace(id="resp_123"))

    async def _wait(response_id: str, **kwargs):
        assert response_id == "resp_123"
        return SimpleNamespace(ok=True, completion=SimpleNamespace(content="final report"))

    provider.clarify_deep_research_task = _clarify  # type: ignore[method-assign]
    provider.rewrite_deep_research_prompt = _rewrite  # type: ignore[method-assign]
    provider.start_deep_research = _start  # type: ignore[method-assign]
    provider.wait_background_response = _wait  # type: ignore[method-assign]

    result = await OpenAIProvider.run_deep_research(
        provider,
        "Research semaglutide",
        clarify_first=True,
        clarifications=["Canada", "2025"],
        rewrite_prompt=True,
        wait_for_completion=True,
        web_search=True,
    )

    assert result.prompt == "Research semaglutide"
    assert result.effective_prompt == "Research semaglutide in Canada during 2025"
    assert result.response_id == "resp_123"
    assert result.clarification.content == "What region and timeframe?"
    assert result.rewrite.content == "Research semaglutide in Canada during 2025"
    assert result.background.completion.content == "final report"
