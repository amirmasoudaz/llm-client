"""
Provider protocol and base classes.

This module defines the abstract interface that all LLM providers must implement,
enabling provider-agnostic agent and application code.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .types import (
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
    EmbeddingResult,
    FileContentResult,
    FileResource,
    FilesPage,
    FineTuningJobEventsPage,
    FineTuningJobResult,
    FineTuningJobsPage,
    ImageGenerationResult,
    Message,
    MessageInput,
    ModerationResult,
    StreamEvent,
    Usage,
    RealtimeCallResult,
    RealtimeConnection,
    RealtimeClientSecretResult,
    RealtimeTranscriptionSessionResult,
    VectorStoreFileBatchResource,
    VectorStoreFileContentResult,
    VectorStoreFileResource,
    VectorStoreFilesPage,
    VectorStoreResource,
    VectorStoreSearchResult,
    VectorStoresPage,
    WebhookEventResult,
    normalize_messages,
)
from ..retry_policy import DEFAULT_RETRYABLE_STATUSES, compute_backoff_delay, extract_retry_after_seconds, is_retryable_status
from ..tools.base import Tool, ToolDefinition

T = TypeVar("T")

if TYPE_CHECKING:
    from ..models import ModelProfile


@runtime_checkable
class Provider(Protocol):
    """
    Protocol defining the interface for LLM providers.

    All providers must implement these core methods to be compatible
    with the agent framework.
    """

    @property
    def model(self) -> type[ModelProfile]:
        """Get the model profile for this provider."""
        ...

    @property
    def model_name(self) -> str:
        """Get the model name string."""
        ...

    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion for the given messages.

        Args:
            messages: Input messages (str, dict, Message, or list of these)
            tools: Optional list of tools the model can call
            tool_choice: How to handle tool selection ("auto", "none", "required", or specific tool)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Output format ("text", "json_object", or JSON schema)
            **kwargs: Provider-specific parameters

        Returns:
            CompletionResult with the model's response
        """
        ...

    def stream(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a completion as a series of events.

        Args:
            messages: Input messages
            tools: Optional list of tools
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response_format: Output format ("text", "json_object", or JSON schema)
            **kwargs: Provider-specific parameters

        Yields:
            StreamEvent objects for tokens, tool calls, usage, etc.
        """
        ...

    async def embed(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for the given inputs.

        Args:
            inputs: Text or list of texts to embed
            **kwargs: Provider-specific parameters

        Returns:
            EmbeddingResult with embedding vectors
        """
        ...

    async def moderate(
        self,
        inputs: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResult:
        """Moderate text or multimodal inputs when supported."""
        ...

    async def generate_image(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate images when supported."""
        ...

    async def edit_image(
        self,
        image: Any,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Edit images when supported."""
        ...

    async def transcribe_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        """Transcribe audio to text when supported."""
        ...

    async def translate_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        """Translate audio to English text when supported."""
        ...

    async def synthesize_speech(
        self,
        text: str,
        *,
        voice: str,
        **kwargs: Any,
    ) -> AudioSpeechResult:
        """Synthesize speech from text when supported."""
        ...

    async def create_file(self, *, file: Any, purpose: str, **kwargs: Any) -> FileResource:
        """Upload a generic provider file when supported."""
        ...

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> FileResource:
        """Retrieve a generic provider file when supported."""
        ...

    async def list_files(self, **kwargs: Any) -> FilesPage:
        """List generic provider files when supported."""
        ...

    async def delete_file(self, file_id: str, **kwargs: Any) -> DeletionResult:
        """Delete a generic provider file when supported."""
        ...

    async def get_file_content(self, file_id: str, **kwargs: Any) -> FileContentResult:
        """Fetch binary content for a generic provider file when supported."""
        ...

    async def create_vector_store(self, **kwargs: Any) -> VectorStoreResource:
        """Create a hosted vector store when supported."""
        ...

    async def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        """Retrieve a hosted vector store when supported."""
        ...

    async def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        """Update a hosted vector store when supported."""
        ...

    async def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> DeletionResult:
        """Delete a hosted vector store when supported."""
        ...

    async def list_vector_stores(self, **kwargs: Any) -> VectorStoresPage:
        """List hosted vector stores when supported."""
        ...

    async def search_vector_store(
        self,
        vector_store_id: str,
        *,
        query: str | list[str],
        **kwargs: Any,
    ) -> VectorStoreSearchResult:
        """Search a hosted vector store when supported."""
        ...

    async def poll_vector_store(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreResource:
        """Poll a hosted vector store until ingestion reaches a terminal state when supported."""
        ...

    async def create_vector_store_and_poll(self, **kwargs: Any) -> VectorStoreResource:
        """Create a hosted vector store and poll until ingestion reaches a terminal state when supported."""
        ...

    async def create_fine_tuning_job(self, **kwargs: Any) -> FineTuningJobResult:
        """Create a fine-tuning job when supported."""
        ...

    async def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        """Retrieve a fine-tuning job when supported."""
        ...

    async def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        """Cancel a fine-tuning job when supported."""
        ...

    async def list_fine_tuning_jobs(self, **kwargs: Any) -> FineTuningJobsPage:
        """List fine-tuning jobs when supported."""
        ...

    async def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> FineTuningJobEventsPage:
        """List fine-tuning job events when supported."""
        ...

    async def create_realtime_client_secret(self, **kwargs: Any) -> RealtimeClientSecretResult:
        """Create a realtime client secret when supported."""
        ...

    async def create_realtime_transcription_session(self, **kwargs: Any) -> RealtimeTranscriptionSessionResult:
        """Create a realtime transcription session when supported."""
        ...

    async def connect_realtime(self, **kwargs: Any) -> RealtimeConnection:
        """Open a realtime connection when supported."""
        ...

    async def connect_realtime_transcription(self, **kwargs: Any) -> RealtimeConnection:
        """Open a realtime transcription connection when supported."""
        ...

    async def create_realtime_call(self, sdp: str, **kwargs: Any) -> RealtimeCallResult:
        """Create a realtime call/session when supported."""
        ...

    async def accept_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        """Accept a realtime call when supported."""
        ...

    async def reject_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        """Reject a realtime call when supported."""
        ...

    async def hangup_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        """Hang up a realtime call when supported."""
        ...

    async def refer_realtime_call(self, call_id: str, *, target_uri: str, **kwargs: Any) -> RealtimeCallResult:
        """Refer a realtime call when supported."""
        ...

    async def unwrap_webhook(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
    ) -> WebhookEventResult:
        """Verify and parse a webhook event when supported."""
        ...

    async def verify_webhook_signature(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
        tolerance: int = 300,
    ) -> bool:
        """Verify a webhook signature when supported."""
        ...

    async def create_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Attach a file to a vector store when supported."""
        ...

    async def upload_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Upload and attach a file to a vector store when supported."""
        ...

    async def list_vector_store_files(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        """List vector-store files when supported."""
        ...

    async def retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Retrieve a vector-store file when supported."""
        ...

    async def update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Update vector-store file metadata when supported."""
        ...

    async def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> DeletionResult:
        """Delete a vector-store file when supported."""
        ...

    async def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileContentResult:
        """Fetch vector-store file content chunks when supported."""
        ...

    async def poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Poll a vector-store file until it reaches a terminal status when supported."""
        ...

    async def create_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Attach a file to a vector store and poll until ready when supported."""
        ...

    async def upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        """Upload a file to a vector store and poll until ready when supported."""
        ...

    async def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Create a vector-store file batch when supported."""
        ...

    async def retrieve_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Retrieve a vector-store file batch when supported."""
        ...

    async def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Cancel a vector-store file batch when supported."""
        ...

    async def poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Poll a vector-store file batch until terminal state when supported."""
        ...

    async def list_vector_store_file_batch_files(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        """List files in a vector-store file batch when supported."""
        ...

    async def create_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Create a vector-store file batch and poll until ready when supported."""
        ...

    async def upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        *,
        files: list[Any] | tuple[Any, ...],
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        """Upload a vector-store file batch and poll until ready when supported."""
        ...

    async def clarify_deep_research_task(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate clarifying questions for a deep-research task when supported."""
        ...

    async def rewrite_deep_research_prompt(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Rewrite a deep-research prompt before kickoff when supported."""
        ...

    async def respond_with_web_search(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted web-search tool when supported."""
        ...

    async def respond_with_file_search(
        self,
        prompt: str,
        *,
        vector_store_ids: list[str] | tuple[str, ...],
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted file-search tool when supported."""
        ...

    async def respond_with_code_interpreter(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted code-interpreter tool when supported."""
        ...

    async def respond_with_shell(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted shell tool when supported."""
        ...

    async def respond_with_apply_patch(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted apply-patch tool when supported."""
        ...

    async def respond_with_computer_use(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted computer-use tool when supported."""
        ...

    async def respond_with_image_generation(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with the hosted image-generation tool when supported."""
        ...

    async def respond_with_remote_mcp(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with a remote MCP server tool when supported."""
        ...

    async def respond_with_connector(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Run a Responses request with a connector-backed MCP tool when supported."""
        ...

    async def start_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Start a deep-research workflow when supported."""
        ...

    async def run_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> DeepResearchRunResult:
        """Run a staged deep-research workflow when supported."""
        ...

    def count_tokens(self, content: Any) -> int:
        """Count tokens in the given content."""
        ...

    def parse_usage(self, raw_usage: dict[str, Any]) -> Usage:
        """Parse raw API usage into Usage object."""
        ...

    async def close(self) -> None:
        """Clean up provider resources."""
        ...

    async def retrieve_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Retrieve a background response by provider-specific identifier."""
        ...

    async def cancel_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Cancel a background response by provider-specific identifier."""
        ...

    def stream_background_response(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Resume or attach to a background response stream when supported."""
        ...

    async def wait_background_response(
        self,
        response_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        """Poll until a background response reaches a terminal state."""
        ...

    async def create_conversation(
        self,
        *,
        items: MessageInput | list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        """Create a provider-native conversation resource when supported."""
        ...

    async def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        """Retrieve a provider-native conversation resource when supported."""
        ...

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        """Update provider-native conversation metadata when supported."""
        ...

    async def delete_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        """Delete a provider-native conversation resource when supported."""
        ...

    async def compact_response_context(
        self,
        *,
        messages: MessageInput | list[dict[str, Any]] | None = None,
        model: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> CompactionResult:
        """Compact a provider-native response window when supported."""
        ...

    async def submit_mcp_approval_response(
        self,
        *,
        previous_response_id: str,
        approval_request_id: str,
        approve: bool,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Submit an MCP approval response and continue the provider workflow."""
        ...

    async def delete_response(self, response_id: str, **kwargs: Any) -> DeletionResult:
        """Delete a stored provider response when supported."""
        ...

    async def create_conversation_items(
        self,
        conversation_id: str,
        *,
        items: MessageInput | list[dict[str, Any]],
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        """Create conversation items within a provider-native conversation."""
        ...

    async def list_conversation_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        order: str | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        """List items within a provider-native conversation."""
        ...

    async def retrieve_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemResource:
        """Retrieve a provider-native conversation item when supported."""
        ...

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        **kwargs: Any,
    ) -> ConversationResource:
        """Delete a provider-native conversation item when supported."""
        ...

    async def __aenter__(self) -> Provider:
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...


class BaseProvider(Provider, ABC):
    """
    Abstract base class for provider implementations.

    Provides common functionality and default implementations while
    requiring subclasses to implement provider-specific methods.
    """

    def __init__(
        self,
        model: type[ModelProfile] | str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            model: ModelProfile class or model key string
            **kwargs: Provider-specific configuration
        """
        from ..models import ModelProfile

        if isinstance(model, str):
            self._model = ModelProfile.get(model)
        elif isinstance(model, type) and issubclass(model, ModelProfile):
            self._model = model
        else:
            raise ValueError("Model must be a ModelProfile subclass or a model key string")

    @property
    def model(self) -> type[ModelProfile]:
        """Get the model profile."""
        return self._model

    @property
    def model_name(self) -> str:
        """Get the model name string."""
        return self._model.model_name

    def count_tokens(self, content: Any) -> int:
        """Count tokens in the given content."""
        return self._model.count_tokens(content)

    def parse_usage(self, raw_usage: dict[str, Any]) -> Usage:
        """Parse raw API usage into Usage object."""
        parsed = self._model.parse_usage(raw_usage)
        return Usage(
            input_tokens=int(parsed.get("input_tokens", 0) or 0),
            output_tokens=int(parsed.get("output_tokens", 0) or 0),
            total_tokens=int(parsed.get("total_tokens", 0) or 0),
            input_tokens_cached=int(parsed.get("input_tokens_cached", 0) or 0),
            output_tokens_reasoning=int(parsed.get("output_tokens_reasoning", 0) or 0),
            input_cost=float(parsed.get("input_cost", 0.0) or 0.0),
            output_cost=float(parsed.get("output_cost", 0.0) or 0.0),
            total_cost=float(parsed.get("total_cost", 0.0) or 0.0),
        )

    @staticmethod
    def _normalize_messages(messages: MessageInput) -> list[Message]:
        """Normalize message input to list of Message objects."""
        return normalize_messages(messages)

    @staticmethod
    def _messages_to_api_format(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to API-compatible format."""
        return [msg.to_dict() for msg in messages]

    @staticmethod
    def _tools_to_api_format(tools: list[ToolDefinition] | None) -> list[dict[str, Any]] | None:
        """Convert Tool objects to API-compatible format."""
        if not tools:
            return None
        out: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                out.append(dict(tool))
                continue
            if hasattr(tool, "to_openai_format"):
                rendered = tool.to_openai_format()
                if isinstance(rendered, dict):
                    out.append(dict(rendered))
                    continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return out

    @staticmethod
    async def _with_retry(
        operation: Callable[[], Any],
        *,
        attempts: int = 3,
        backoff: float = 1.0,
        retryable_statuses: tuple[int, ...] = DEFAULT_RETRYABLE_STATUSES,
    ) -> Any:
        """
        Execute an operation with retry, exponential backoff, and Retry-After support.

        Args:
            operation: Async callable to execute
            attempts: Maximum number of attempts
            backoff: Initial backoff delay in seconds (doubles each retry)
            retryable_statuses: HTTP status codes that should trigger retry

        Returns:
            Result of the operation

        Note:
            The operation should return a result with a 'status' attribute
            or raise an exception on failure.
        """
        last_result = None
        last_error = None
        current_backoff = backoff

        for attempt in range(attempts):
            try:
                result = await operation()

                # Check if result indicates a retryable error
                if hasattr(result, "status"):
                    if is_retryable_status(result.status, retryable_statuses=retryable_statuses) and attempt < attempts - 1:
                        last_result = result
                        wait_time = compute_backoff_delay(
                            attempt=attempt,
                            base_backoff=current_backoff,
                            retry_after=extract_retry_after_seconds(result),
                        )

                        await asyncio.sleep(wait_time)
                        current_backoff *= 2
                        continue

                return result

            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(
                        compute_backoff_delay(
                            attempt=attempt,
                            base_backoff=current_backoff,
                            retry_after=extract_retry_after_seconds(e),
                        )
                    )
                    current_backoff *= 2
                    continue
                raise

        # Return last result if we have one, otherwise raise last error
        if last_result is not None:
            return last_result
        if last_error is not None:
            raise last_error

        raise RuntimeError("Retry logic failed unexpectedly")

    @abstractmethod
    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion. Must be implemented by subclasses."""
        ...

    async def moderate(
        self,
        inputs: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support moderation.")

    async def generate_image(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support image generation.")

    async def edit_image(
        self,
        image: Any,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support image editing.")

    async def transcribe_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support speech-to-text.")

    async def translate_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support audio translation.")

    async def synthesize_speech(
        self,
        text: str,
        *,
        voice: str,
        **kwargs: Any,
    ) -> AudioSpeechResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not support text-to-speech.")

    async def create_file(self, *, file: Any, purpose: str, **kwargs: Any) -> FileResource:
        _ = file, purpose, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support generic file uploads.")

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> FileResource:
        _ = file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support generic files.")

    async def list_files(self, **kwargs: Any) -> FilesPage:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support generic files.")

    async def delete_file(self, file_id: str, **kwargs: Any) -> DeletionResult:
        _ = file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support generic files.")

    async def get_file_content(self, file_id: str, **kwargs: Any) -> FileContentResult:
        _ = file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support generic file content retrieval.")

    async def create_vector_store(self, **kwargs: Any) -> VectorStoreResource:
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector stores.")

    async def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector stores.")

    async def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector stores.")

    async def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> DeletionResult:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector stores.")

    async def list_vector_stores(self, **kwargs: Any) -> VectorStoresPage:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector stores.")

    async def search_vector_store(
        self,
        vector_store_id: str,
        *,
        query: str | list[str],
        **kwargs: Any,
    ) -> VectorStoreSearchResult:
        _ = vector_store_id, query, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store search.")

    @staticmethod
    def _vector_store_ingestion_terminal(result: VectorStoreResource) -> bool:
        terminal_statuses = {"completed", "failed", "cancelled", "expired"}
        if str(result.status or "").lower() in terminal_statuses:
            return True
        file_counts = result.file_counts or {}
        in_progress = file_counts.get("in_progress")
        return isinstance(in_progress, int) and in_progress == 0

    async def poll_vector_store(
        self,
        vector_store_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreResource:
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        while True:
            result = await self.retrieve_vector_store(vector_store_id, **kwargs)
            if self._vector_store_ingestion_terminal(result):
                return result
            if timeout is not None and (loop.time() - started_at) >= timeout:
                raise TimeoutError(f"Timed out waiting for vector store {vector_store_id!r}")
            await asyncio.sleep(poll_interval)

    async def create_vector_store_and_poll(
        self,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreResource:
        initial_file_ids = kwargs.get("file_ids")
        result = await self.create_vector_store(**kwargs)
        if not initial_file_ids:
            return result
        return await self.poll_vector_store(
            result.vector_store_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def create_fine_tuning_job(self, **kwargs: Any) -> FineTuningJobResult:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning.")

    async def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        _ = job_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning.")

    async def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        _ = job_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning.")

    async def list_fine_tuning_jobs(self, **kwargs: Any) -> FineTuningJobsPage:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning.")

    async def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> FineTuningJobEventsPage:
        _ = job_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning.")

    async def create_realtime_client_secret(self, **kwargs: Any) -> RealtimeClientSecretResult:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def connect_realtime(self, **kwargs: Any) -> RealtimeConnection:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def create_realtime_transcription_session(self, **kwargs: Any) -> RealtimeTranscriptionSessionResult:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime transcription.")

    async def connect_realtime_transcription(self, **kwargs: Any) -> RealtimeConnection:
        _ = kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime transcription.")

    async def create_realtime_call(self, sdp: str, **kwargs: Any) -> RealtimeCallResult:
        _ = sdp, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def accept_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        _ = call_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def reject_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        _ = call_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def hangup_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        _ = call_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def refer_realtime_call(self, call_id: str, *, target_uri: str, **kwargs: Any) -> RealtimeCallResult:
        _ = call_id, target_uri, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support realtime.")

    async def unwrap_webhook(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
    ) -> WebhookEventResult:
        _ = payload, headers, secret
        raise NotImplementedError(f"{self.__class__.__name__} does not support webhooks.")

    async def verify_webhook_signature(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
        tolerance: int = 300,
    ) -> bool:
        _ = payload, headers, secret, tolerance
        raise NotImplementedError(f"{self.__class__.__name__} does not support webhooks.")

    async def create_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def upload_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def list_vector_store_files(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> DeletionResult:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileContentResult:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store files.")

    async def poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file polling.")

    async def create_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file polling.")

    async def upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        _ = vector_store_id, file, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file polling.")

    async def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def retrieve_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, batch_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, batch_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, batch_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def list_vector_store_file_batch_files(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        _ = vector_store_id, batch_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def create_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        *,
        files: list[Any] | tuple[Any, ...],
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        _ = vector_store_id, files, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector-store file batches.")

    async def clarify_deep_research_task(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support deep-research clarification.")

    async def rewrite_deep_research_prompt(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support deep-research prompt rewriting.")

    async def respond_with_web_search(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted web-search workflows.")

    async def respond_with_file_search(
        self,
        prompt: str,
        *,
        vector_store_ids: list[str] | tuple[str, ...],
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, vector_store_ids, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted file-search workflows.")

    async def respond_with_code_interpreter(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted code-interpreter workflows.")

    async def respond_with_shell(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted shell workflows.")

    async def respond_with_apply_patch(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted apply-patch workflows.")

    async def respond_with_computer_use(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted computer-use workflows.")

    async def respond_with_image_generation(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support hosted image-generation workflows.")

    async def respond_with_remote_mcp(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support remote MCP workflows.")

    async def respond_with_connector(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support connector workflows.")

    async def start_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support deep-research orchestration.")

    async def run_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> DeepResearchRunResult:
        _ = prompt, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} does not support staged deep-research orchestration.")

    async def complete_structured(
        self,
        messages: MessageInput,
        schema: dict[str, Any],
        *,
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> Any:  # Returns StructuredResult
        """
        Complete with structured output validation.
        
        Automatically validates output against the schema and attempts
        repair if validation fails.
        
        Args:
            messages: Input messages
            schema: JSON schema for the expected output
            max_repair_attempts: Max repair attempts (default: 2)
            **kwargs: Additional arguments for complete()
        
        Returns:
            StructuredResult with validated data
        """
        from ..structured import StructuredOutputConfig, structured

        normalized = self._normalize_messages(messages)
        config = StructuredOutputConfig(
            schema=schema,
            max_repair_attempts=max_repair_attempts,
        )
        return await structured(
            provider=self,
            messages=normalized,
            config=config,
            mode="complete",
            **kwargs,
        )

    def stream_structured(
        self,
        messages: MessageInput,
        schema: dict[str, Any],
        *,
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream structured output using the unified structured dispatcher."""
        from ..structured import StructuredOutputConfig, structured

        normalized = self._normalize_messages(messages)
        config = StructuredOutputConfig(
            schema=schema,
            max_repair_attempts=max_repair_attempts,
        )
        return structured(
            provider=self,
            messages=normalized,
            config=config,
            mode="stream",
            **kwargs,
        )

    async def embed(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings.

        Default implementation raises NotImplementedError.
        Override in subclasses that support embeddings.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    async def close(self) -> None:
        """
        Clean up provider resources.

        Override in subclasses that need cleanup.
        """
        pass

    async def retrieve_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Retrieve a background response when supported by the provider."""
        _ = (response_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support background response retrieval")

    async def cancel_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Cancel a background response when supported by the provider."""
        _ = (response_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support background response cancellation")

    def stream_background_response(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Resume or attach to a background response stream when supported."""
        _ = (response_id, starting_after, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support background response streaming")

    async def wait_background_response(
        self,
        response_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        """Poll a background response until it reaches a terminal lifecycle state."""
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        while True:
            result = await self.retrieve_background_response(response_id, **kwargs)
            if result.is_terminal:
                return result
            if timeout is not None and (loop.time() - started_at) >= timeout:
                raise TimeoutError(f"Timed out waiting for background response {response_id!r}")
            await asyncio.sleep(poll_interval)

    async def create_conversation(
        self,
        *,
        items: MessageInput | list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        _ = (items, metadata, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation creation")

    async def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        _ = (conversation_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation retrieval")

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        _ = (conversation_id, metadata, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation updates")

    async def delete_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        _ = (conversation_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation deletion")

    async def compact_response_context(
        self,
        *,
        messages: MessageInput | list[dict[str, Any]] | None = None,
        model: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> CompactionResult:
        _ = (messages, model, instructions, previous_response_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support response compaction")

    async def submit_mcp_approval_response(
        self,
        *,
        previous_response_id: str,
        approval_request_id: str,
        approve: bool,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        _ = (previous_response_id, approval_request_id, approve, tools, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support MCP approval flows")

    async def delete_response(self, response_id: str, **kwargs: Any) -> DeletionResult:
        _ = (response_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support stored response deletion")

    async def create_conversation_items(
        self,
        conversation_id: str,
        *,
        items: MessageInput | list[dict[str, Any]],
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        _ = (conversation_id, items, include, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation item creation")

    async def list_conversation_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        order: str | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        _ = (conversation_id, after, include, limit, order, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation item listing")

    async def retrieve_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemResource:
        _ = (conversation_id, item_id, include, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation item retrieval")

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        **kwargs: Any,
    ) -> ConversationResource:
        _ = (conversation_id, item_id, kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversation item deletion")

    async def __aenter__(self) -> Provider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = [
    "Provider",
    "BaseProvider",
]
