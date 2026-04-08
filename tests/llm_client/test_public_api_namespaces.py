import llm_client
import warnings

from llm_client import adapters, budgets, compat, config, content, engine, models, observability, providers, tools, types, validation
from llm_client.compat import OpenAIClient, ResponseTimeoutError
from llm_client.adapters import PostgresSQLAdaptor, SQLQueryRequest
from llm_client.content import Message, Role, ToolCall, normalize_messages
from llm_client.budgets import Budget, Ledger
from llm_client.observability import (
    BenchmarkRecorder,
    EngineDiagnosticsRecorder,
    EngineDiagnosticsSnapshot,
    InMemoryEventBus,
    LatencyRecorder,
    LifecycleLoggingHook,
    ReplayRecorder,
    RedactionPolicy,
    RuntimeEvent,
    RuntimeEventType,
)
from llm_client.tools import (
    ResponsesAttributeFilter,
    ResponsesBuiltinTool,
    ResponsesChunkingStrategy,
    ResponsesConnectorId,
    ResponsesCustomTool,
    ResponsesDropboxTool,
    ResponsesExpirationPolicy,
    ResponsesFileSearchHybridWeights,
    ResponsesFileSearchRankingOptions,
    ResponsesFunctionTool,
    ResponsesGmailTool,
    ResponsesGoogleCalendarTool,
    ResponsesGoogleDriveTool,
    ResponsesGrammar,
    ResponsesMCPApprovalPolicy,
    ResponsesMCPTool,
    ResponsesMCPToolFilter,
    ResponsesMicrosoftTeamsTool,
    ResponsesOutlookCalendarTool,
    ResponsesOutlookEmailTool,
    ResponsesSharePointTool,
    ResponsesToolNamespace,
    ResponsesToolSearch,
    ResponsesVectorStoreFileSpec,
)
from llm_client.types import (
    AudioSpeechResult,
    AudioTranscriptionResult,
    BackgroundResponseResult,
    CancellationToken,
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
    FineTuningJobsPage,
    GeneratedImage,
    ImageGenerationResult,
    ModerationResult,
    NormalizedOutputItem,
    RealtimeCallResult,
    RealtimeConnection,
    RealtimeClientSecretResult,
    RealtimeEventResult,
    RealtimeTranscriptionSessionResult,
    RequestContext,
    RequestSpec,
    StreamEvent,
    StreamEventType,
    Usage,
    VectorStoreFileContentResult,
    VectorStoreFileBatchResource,
    VectorStoreFileResource,
    VectorStoreFilesPage,
    VectorStoreResource,
    VectorStoreSearchResult,
    VectorStoresPage,
    WebhookEventResult,
)


def test_content_namespace_exports_core_message_types() -> None:
    tool_call = ToolCall(id="call_1", name="lookup", arguments='{"q":"hello"}')
    message = Message.assistant(tool_calls=[tool_call])

    assert message.role is Role.ASSISTANT
    assert normalize_messages([message])[0].tool_calls[0].name == "lookup"


def test_types_namespace_exports_shared_request_and_result_types() -> None:
    spec = RequestSpec(
        provider="openai",
        model="gpt-5",
        messages=[Message.user("hi")],
    )
    result = CompletionResult(content="hello", usage=Usage(total_tokens=3))
    background = BackgroundResponseResult(response_id="resp_1", lifecycle_status="queued")
    conversation = ConversationResource(conversation_id="conv_1")
    compaction = CompactionResult(compaction_id="cmp_1")
    deletion = DeletionResult(resource_id="resp_2")
    file_resource = FileResource(file_id="file_1", filename="guide.pdf", purpose="assistants")
    files_page = FilesPage(items=[file_resource])
    file_content = FileContentResult(file_id="file_1", content=b"abc", media_type="application/pdf")
    conversation_item = ConversationItemResource(item_id="item_1", item_type="message")
    conversation_items = ConversationItemsPage(items=[conversation_item])
    moderation = ModerationResult(flagged=False, model="omni-moderation-latest")
    image = ImageGenerationResult(images=[GeneratedImage(url="https://example.com/image.png")], model="gpt-image-1")
    transcript = AudioTranscriptionResult(text="hello", language="en", model="gpt-4o-transcribe")
    speech = AudioSpeechResult(audio=b"abc", format="mp3", model="tts-1")
    vector_store = VectorStoreResource(vector_store_id="vs_1", name="Docs")
    vector_stores = VectorStoresPage(items=[vector_store])
    vector_store_file = VectorStoreFileResource(file_id="file_1", vector_store_id="vs_1")
    vector_store_files = VectorStoreFilesPage(items=[vector_store_file])
    vector_store_content = VectorStoreFileContentResult(file_id="file_1", vector_store_id="vs_1", chunks=[{"text": "hello"}])
    vector_search = VectorStoreSearchResult(vector_store_id="vs_1", query="docs", results=[{"file_id": "file_1"}])
    fine_tune = FineTuningJobResult(job_id="ftjob_1", status="queued")
    fine_tune_jobs = FineTuningJobsPage(items=[fine_tune])
    fine_tune_events = FineTuningJobEventsPage(job_id="ftjob_1", events=[{"id": "evt_1"}])
    realtime_secret = RealtimeClientSecretResult(value="secret_1")
    realtime_transcription = RealtimeTranscriptionSessionResult(value="tx_secret_1")
    realtime_call = RealtimeCallResult(call_id="rtc_1", action="create")
    realtime_event = RealtimeEventResult(event_type="conversation.item.added", event_id="evt_1")
    realtime_connection = RealtimeConnection(connection=object(), model="gpt-realtime", call_id="rtc_1")
    webhook = WebhookEventResult(event_id="wh_1", event_type="realtime.call.incoming", data={"call_id": "rtc_1"})
    vector_store_batch = VectorStoreFileBatchResource(batch_id="vsfb_1", vector_store_id="vs_1", status="in_progress")
    output_item = NormalizedOutputItem(type="refusal", text="No")
    event = StreamEvent(type=StreamEventType.DONE, data=result)
    research_run = DeepResearchRunResult(prompt="Research", effective_prompt="Research", queued=result)

    assert spec.provider == "openai"
    assert RequestContext(session_id="session-1").session_id == "session-1"
    assert result.usage.total_tokens == 3
    assert background.lifecycle_status == "queued"
    assert conversation.conversation_id == "conv_1"
    assert compaction.compaction_id == "cmp_1"
    assert deletion.resource_id == "resp_2"
    assert file_resource.filename == "guide.pdf"
    assert files_page.items[0].purpose == "assistants"
    assert file_content.byte_length == 3
    assert conversation_item.item_id == "item_1"
    assert conversation_items.items[0].item_type == "message"
    assert moderation.model == "omni-moderation-latest"
    assert image.images[0].url == "https://example.com/image.png"
    assert transcript.language == "en"
    assert speech.byte_length == 3
    assert vector_store.vector_store_id == "vs_1"
    assert vector_stores.items[0].name == "Docs"
    assert vector_store_file.file_id == "file_1"
    assert vector_store_files.items[0].vector_store_id == "vs_1"
    assert vector_store_content.chunks[0]["text"] == "hello"
    assert vector_search.results[0]["file_id"] == "file_1"
    assert fine_tune.job_id == "ftjob_1"
    assert fine_tune_jobs.items[0].status == "queued"
    assert fine_tune_events.events[0]["id"] == "evt_1"
    assert realtime_secret.value == "secret_1"
    assert realtime_transcription.value == "tx_secret_1"
    assert realtime_call.call_id == "rtc_1"
    assert realtime_event.event_type == "conversation.item.added"
    assert realtime_connection.call_id == "rtc_1"
    assert webhook.event_type == "realtime.call.incoming"
    assert vector_store_batch.batch_id == "vsfb_1"
    assert output_item.type == "refusal"
    assert event.type.value == "done"
    assert research_run.effective_prompt == "Research"
    assert isinstance(CancellationToken.none(), CancellationToken)


def test_observability_namespace_exports_runtime_and_replay_primitives() -> None:
    bus = InMemoryEventBus()
    recorder = ReplayRecorder(event_bus=bus)
    runtime_event = RuntimeEvent(type=RuntimeEventType.PROGRESS, data={"step": "start"})
    diagnostics = EngineDiagnosticsSnapshot(kind="request", request_id="req-1", payload={"status": 200})

    assert recorder.is_recording is False
    assert runtime_event.type is RuntimeEventType.PROGRESS
    assert isinstance(LatencyRecorder(), LatencyRecorder)
    assert isinstance(EngineDiagnosticsRecorder(), EngineDiagnosticsRecorder)
    assert isinstance(BenchmarkRecorder(), BenchmarkRecorder)
    assert isinstance(LifecycleLoggingHook(), LifecycleLoggingHook)
    assert isinstance(RedactionPolicy(), RedactionPolicy)
    assert diagnostics.request_id == "req-1"


def test_tools_namespace_exports_vector_store_resource_helpers() -> None:
    expiration = ResponsesExpirationPolicy(days=7)
    chunking = ResponsesChunkingStrategy.static(max_chunk_size_tokens=1200, chunk_overlap_tokens=200)
    file_spec = ResponsesVectorStoreFileSpec(
        file_id="file_1",
        attributes={"scope": "tenant"},
        chunking_strategy=ResponsesChunkingStrategy.auto(),
    )

    assert expiration.to_dict() == {"anchor": "last_active_at", "days": 7}
    assert chunking.to_dict() == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 1200, "chunk_overlap_tokens": 200},
    }
    assert file_spec.to_dict() == {
        "file_id": "file_1",
        "attributes": {"scope": "tenant"},
        "chunking_strategy": {"type": "auto"},
    }


def test_compat_namespace_exposes_legacy_client_api() -> None:
    assert OpenAIClient is not None
    assert issubclass(ResponseTimeoutError, TimeoutError)


def test_top_level_compat_aliases_warn_and_resolve() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_client = llm_client.OpenAIClient

    assert legacy_client is OpenAIClient
    assert any("compatibility-only" in str(item.message) for item in caught)


def test_top_level_all_is_now_a_smaller_stable_contract() -> None:
    assert "Agent" in llm_client.__all__
    assert "OpenAIProvider" in llm_client.__all__
    assert "OpenAIClient" in llm_client.__all__
    assert "TelemetryCounter" not in llm_client.__all__
    assert "fingerprint" not in llm_client.__all__
    assert not hasattr(llm_client, "TelemetryCounter")
    assert not hasattr(llm_client, "fingerprint")
    assert not hasattr(llm_client, "BatchManager")


def test_public_modules_define_explicit_all_contracts() -> None:
    assert "SQLQueryRequest" in adapters.__all__
    assert "ExecutionEngine" in engine.__all__
    assert "Ledger" in budgets.__all__
    assert "Message" in content.__all__
    assert "OpenAIProvider" in providers.__all__
    assert "ToolRegistry" in tools.__all__
    assert "RequestContext" in types.__all__
    assert "RuntimeEvent" in observability.__all__
    assert "ValidationError" in validation.__all__
    assert "load_env" in config.__all__
    assert "GPT5" in models.__all__
    assert "OpenAIClient" in compat.__all__


def test_tools_namespace_exports_responses_tool_descriptors() -> None:
    builtin = ResponsesBuiltinTool.web_search(search_context_size="low")
    attribute_filter = ResponsesAttributeFilter.and_(
        ResponsesAttributeFilter.eq("scope", "tenant"),
        ResponsesAttributeFilter.gte("score", 0.8),
    )
    ranking = ResponsesFileSearchRankingOptions(
        ranker="default-2024-11-15",
        score_threshold=0.25,
        hybrid_search=ResponsesFileSearchHybridWeights(embedding_weight=0.7, text_weight=0.3),
    )
    tool_search = ResponsesToolSearch.client(parameters={"type": "object", "properties": {"query": {"type": "string"}}})
    custom = ResponsesCustomTool(
        name="planner",
        description="Emit a plan.",
        grammar=ResponsesGrammar(syntax="lark", definition='start: "done"'),
    )
    function_tool = ResponsesFunctionTool(
        name="lookup_profile",
        description="Lookup a profile.",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        defer_loading=True,
    )
    namespace = ResponsesToolNamespace(
        name="crm",
        description="CRM tools",
        tools=(function_tool,),
    )
    mcp = ResponsesMCPTool.connector(
        ResponsesConnectorId.GMAIL,
        allowed_tools=(ResponsesGmailTool.SEARCH_EMAILS, ResponsesGmailTool.READ_EMAIL),
        defer_loading=True,
        require_approval=ResponsesMCPApprovalPolicy(
            always=ResponsesMCPToolFilter.of(ResponsesGmailTool.READ_EMAIL),
        ),
    )

    assert builtin.to_dict()["type"] == "web_search"
    assert attribute_filter.to_dict()["filters"][0]["type"] == "eq"
    assert ranking.to_dict()["hybrid_search"]["embedding_weight"] == 0.7
    assert tool_search.to_dict()["type"] == "tool_search"
    assert custom.to_dict()["format"]["syntax"] == "lark"
    assert function_tool.to_dict()["defer_loading"] is True
    assert namespace.to_dict()["tools"][0]["name"] == "lookup_profile"
    assert mcp.to_dict()["connector_id"] == "connector_gmail"
    assert mcp.to_dict()["allowed_tools"] == ["search_emails", "read_email"]
    assert mcp.to_dict()["defer_loading"] is True
    assert mcp.to_dict()["require_approval"] == {"always": {"tool_names": ["read_email"]}}
    assert "ResponsesBuiltinTool" in tools.__all__
    assert "ResponsesAttributeFilter" in tools.__all__
    assert "ResponsesToolSearch" in tools.__all__
    assert "ResponsesConnectorId" in tools.__all__
    assert "ResponsesDropboxTool" in tools.__all__
    assert "ResponsesFileSearchHybridWeights" in tools.__all__
    assert "ResponsesFileSearchRankingOptions" in tools.__all__
    assert "ResponsesFunctionTool" in tools.__all__
    assert "ResponsesGmailTool" in tools.__all__
    assert "ResponsesGoogleCalendarTool" in tools.__all__
    assert "ResponsesGoogleDriveTool" in tools.__all__
    assert "ResponsesMicrosoftTeamsTool" in tools.__all__
    assert "ResponsesToolNamespace" in tools.__all__
    assert "ResponsesMCPTool" in tools.__all__
    assert "ResponsesMCPApprovalPolicy" in tools.__all__
    assert "ResponsesMCPToolFilter" in tools.__all__
    assert "ResponsesOutlookCalendarTool" in tools.__all__
    assert "ResponsesOutlookEmailTool" in tools.__all__
    assert "ResponsesSharePointTool" in tools.__all__
    assert "ResponsesCustomTool" in tools.__all__
    assert "ResponsesGrammar" in tools.__all__


def test_budget_namespace_exports_generic_usage_and_budget_primitives() -> None:
    assert Ledger is not None
    assert Budget is not None


def test_adapter_namespace_exports_generic_service_contracts() -> None:
    request = SQLQueryRequest(statement="select 1")

    assert request.statement == "select 1"
    assert PostgresSQLAdaptor.backend_name == "postgres"


def test_public_module_all_contracts_avoid_internal_helpers() -> None:
    assert "_TOOL_SCHEMA_STRICT" not in validation.__all__
    assert "_default_agent_config" not in config.__all__
    assert "_DEFAULT_MAX_TOOL_OUTPUT_CHARS" not in tools.__all__
