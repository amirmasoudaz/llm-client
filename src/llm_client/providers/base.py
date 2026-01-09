"""
Provider protocol and base classes.

This module defines the abstract interface that all LLM providers must implement,
enabling provider-agnostic agent and application code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

from .types import (
    CompletionResult,
    EmbeddingResult,
    Message,
    MessageInput,
    StreamEvent,
    Usage,
    normalize_messages,
)

if TYPE_CHECKING:
    from ..models import ModelProfile
    from ..tools.base import Tool


@runtime_checkable
class Provider(Protocol):
    """
    Protocol defining the interface for LLM providers.
    
    All providers must implement these core methods to be compatible
    with the agent framework.
    """
    
    @property
    def model(self) -> Type["ModelProfile"]:
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
        tools: Optional[List["Tool"]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[str, Dict[str, Any], Type]] = None,
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
    
    async def stream(
        self,
        messages: MessageInput,
        *,
        tools: Optional[List["Tool"]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
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
            **kwargs: Provider-specific parameters
            
        Yields:
            StreamEvent objects for tokens, tool calls, usage, etc.
        """
        ...
    
    async def embed(
        self,
        inputs: Union[str, List[str]],
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


class BaseProvider(ABC):
    """
    Abstract base class for provider implementations.
    
    Provides common functionality and default implementations while
    requiring subclasses to implement provider-specific methods.
    """
    
    def __init__(
        self,
        model: Union[Type["ModelProfile"], str],
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
            raise ValueError(
                "Model must be a ModelProfile subclass or a model key string"
            )
    
    @property
    def model(self) -> Type["ModelProfile"]:
        """Get the model profile."""
        return self._model
    
    @property
    def model_name(self) -> str:
        """Get the model name string."""
        return self._model.model_name
    
    def count_tokens(self, content: Any) -> int:
        """Count tokens in the given content."""
        return self._model.count_tokens(content)
    
    def parse_usage(self, raw_usage: Dict[str, Any]) -> Usage:
        """Parse raw API usage into Usage object."""
        parsed = self._model.parse_usage(raw_usage)
        return Usage(
            input_tokens=parsed.get("input_tokens", 0),
            output_tokens=parsed.get("output_tokens", 0),
            total_tokens=parsed.get("total_tokens", 0),
            input_tokens_cached=parsed.get("input_tokens_cached", 0),
            input_cost=parsed.get("input_cost", 0.0),
            output_cost=parsed.get("output_cost", 0.0),
            total_cost=parsed.get("total_cost", 0.0),
        )
    
    def _normalize_messages(self, messages: MessageInput) -> List[Message]:
        """Normalize message input to list of Message objects."""
        return normalize_messages(messages)
    
    def _messages_to_api_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to API-compatible format."""
        return [msg.to_dict() for msg in messages]
    
    def _tools_to_api_format(self, tools: Optional[List["Tool"]]) -> Optional[List[Dict[str, Any]]]:
        """Convert Tool objects to API-compatible format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in tools
        ]
    
    @abstractmethod
    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: Optional[List["Tool"]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[str, Dict[str, Any], Type]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion. Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    async def stream(
        self,
        messages: MessageInput,
        *,
        tools: Optional[List["Tool"]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion. Must be implemented by subclasses."""
        ...
    
    async def embed(
        self,
        inputs: Union[str, List[str]],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings.
        
        Default implementation raises NotImplementedError.
        Override in subclasses that support embeddings.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )
    
    async def close(self) -> None:
        """
        Clean up provider resources.
        
        Override in subclasses that need cleanup.
        """
        pass
    
    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = [
    "Provider",
    "BaseProvider",
]

