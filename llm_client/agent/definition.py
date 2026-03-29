"""
Immutable agent definition and runtime-state types.

These types separate the static description of an agent from the mutable state
used while the agent is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import AgentConfig


class ToolExecutionMode(str, Enum):
    SINGLE = "single"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PLANNER = "planner"


@dataclass(frozen=True)
class PromptTemplateReference:
    """
    Reference to an attached prompt/template resource.

    A reference may point to an external URI/path or contain an inline template.
    """

    name: str
    uri: str | None = None
    inline_text: str | None = None
    description: str | None = None
    variables: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("PromptTemplateReference.name cannot be empty")
        if not self.uri and not self.inline_text:
            raise ValueError("PromptTemplateReference requires either uri or inline_text")


@dataclass(frozen=True)
class AgentExecutionPolicy:
    max_turns: int = 10
    max_tool_calls_per_turn: int = 10
    tool_execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL
    tool_timeout: float = 30.0
    tool_retry_attempts: int = 0
    batch_concurrency: int = 20
    stop_on_tool_error: bool = False
    include_tool_errors_in_context: bool = True
    stream_tool_calls: bool = True
    use_default_middleware: bool = False

    def to_agent_config_kwargs(self) -> dict[str, Any]:
        # Current agent runtime supports parallel vs non-parallel execution only.
        parallel_tool_execution = self.tool_execution_mode is ToolExecutionMode.PARALLEL
        return {
            "max_turns": self.max_turns,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "parallel_tool_execution": parallel_tool_execution,
            "tool_timeout": self.tool_timeout,
            "tool_retry_attempts": self.tool_retry_attempts,
            "batch_concurrency": self.batch_concurrency,
            "stop_on_tool_error": self.stop_on_tool_error,
            "include_tool_errors_in_context": self.include_tool_errors_in_context,
            "stream_tool_calls": self.stream_tool_calls,
            "use_default_middleware": self.use_default_middleware,
        }


@dataclass(frozen=True)
class AgentOutputPolicy:
    max_tool_output_chars: int | None = None
    include_turns: bool = True
    include_usage: bool = True


@dataclass(frozen=True)
class AgentMemoryPolicy:
    max_tokens: int | None = None
    reserve_tokens: int = 2000
    summarization_enabled: bool = False
    persistent_summary: bool = False
    retrieval_enabled: bool = False


@dataclass(frozen=True)
class AgentDefinition:
    name: str | None = None
    system_message: str | None = None
    prompt_templates: tuple[PromptTemplateReference, ...] = ()
    execution_policy: AgentExecutionPolicy = field(default_factory=AgentExecutionPolicy)
    output_policy: AgentOutputPolicy = field(default_factory=AgentOutputPolicy)
    memory_policy: AgentMemoryPolicy = field(default_factory=AgentMemoryPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_agent_config(self, *, middleware_chain: Any | None = None) -> AgentConfig:
        kwargs = self.execution_policy.to_agent_config_kwargs()
        kwargs.update(
            {
                "max_tool_output_chars": self.output_policy.max_tool_output_chars,
                "max_tokens": self.memory_policy.max_tokens,
                "reserve_tokens": self.memory_policy.reserve_tokens,
                "middleware_chain": middleware_chain,
            }
        )
        return AgentConfig(**kwargs)

    @classmethod
    def from_agent_config(
        cls,
        config: AgentConfig,
        *,
        name: str | None = None,
        system_message: str | None = None,
        prompt_templates: tuple[PromptTemplateReference, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> AgentDefinition:
        tool_mode = ToolExecutionMode.PARALLEL if config.parallel_tool_execution else ToolExecutionMode.SEQUENTIAL
        return cls(
            name=name,
            system_message=system_message,
            prompt_templates=prompt_templates,
            execution_policy=AgentExecutionPolicy(
                max_turns=config.max_turns,
                max_tool_calls_per_turn=config.max_tool_calls_per_turn,
                tool_execution_mode=tool_mode,
                tool_timeout=config.tool_timeout,
                tool_retry_attempts=config.tool_retry_attempts,
                batch_concurrency=config.batch_concurrency,
                stop_on_tool_error=config.stop_on_tool_error,
                include_tool_errors_in_context=config.include_tool_errors_in_context,
                stream_tool_calls=config.stream_tool_calls,
                use_default_middleware=config.use_default_middleware,
            ),
            output_policy=AgentOutputPolicy(
                max_tool_output_chars=config.max_tool_output_chars,
            ),
            memory_policy=AgentMemoryPolicy(
                max_tokens=config.max_tokens,
                reserve_tokens=config.reserve_tokens,
            ),
            metadata=dict(metadata or {}),
        )


@dataclass
class AgentRuntimeState:
    conversation: Any
    request_context: Any | None = None


__all__ = [
    "ToolExecutionMode",
    "PromptTemplateReference",
    "AgentExecutionPolicy",
    "AgentOutputPolicy",
    "AgentMemoryPolicy",
    "AgentDefinition",
    "AgentRuntimeState",
]
