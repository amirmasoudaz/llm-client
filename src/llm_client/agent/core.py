"""
Agent core implementation.

This module provides the main Agent class for autonomous LLM interactions.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from ..cancellation import CancelledError
from ..config import AgentConfig
from ..conversation import Conversation
from ..engine import ExecutionEngine, RetryConfig
from ..providers.base import Provider
from ..providers.types import (
    CompletionResult,
    Message,
    StreamEvent,
    StreamEventType,
    Usage,
)
from ..spec import RequestContext, RequestSpec
from ..streaming import BufferingAdapter
from ..tools.base import Tool, ToolRegistry, ToolResult
from .execution import apply_tool_output_limit, execute_tools
from .result import AgentResult, TurnResult
from .session import load_agent_session, save_agent_session

if TYPE_CHECKING:
    from ..models import ModelProfile


class Agent:
    """
    Autonomous agent with tool calling and multi-turn reasoning.

    The Agent orchestrates:
    - Multi-turn conversations with context management
    - Tool/function calling with automatic execution
    - ReAct-style reasoning loops
    - Streaming responses with tool call events

    Example:
        ```python
        from llm_client import Agent, OpenAIProvider, tool

        @tool
        async def search_web(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"

        agent = Agent(
            provider=OpenAIProvider(model="gpt-5"),
            tools=[search_web],
            system_message="You are a helpful research assistant.",
        )

        # Run agent to completion
        result = await agent.run("What's the weather in NYC?")
        print(result.content)

        # Or stream the response
        async for event in agent.stream("Tell me about Python"):
            if event.type == StreamEventType.TOKEN:
                print(event.data, end="")
        ```
    """

    def __init__(
        self,
        provider: Provider,
        *,
        tools: list[Tool] | ToolRegistry | None = None,
        system_message: str | None = None,
        conversation: Conversation | None = None,
        config: AgentConfig | None = None,
        engine: ExecutionEngine | None = None,
        use_engine: bool = False,
        # Shortcuts for common config options
        max_turns: int = 10,
        max_tokens: int | None = None,
        # Middleware support
        use_middleware: bool = False,
    ) -> None:
        """
        Initialize the agent.

        Args:
            provider: LLM provider for completions
            tools: Tools available to the agent (list or registry)
            system_message: System instruction for the agent
            conversation: Existing conversation to continue
            config: Full agent configuration
            max_turns: Maximum turns (shortcut for config.max_turns)
            max_tokens: Maximum context tokens (shortcut)
            use_middleware: Enable production middleware for tool execution
        """
        self.provider = provider
        self.engine: ExecutionEngine | None = (
            engine if engine is not None else ExecutionEngine(provider=self.provider) if use_engine else None
        )

        # Set up tool registry
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        elif tools:
            self.tools = ToolRegistry(tools)
        else:
            self.tools = ToolRegistry()

        # Set up configuration
        self.config = config or AgentConfig()
        if max_turns != 10:
            self.config.max_turns = max_turns
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        if use_middleware:
            self.config.use_default_middleware = True

        # Set up conversation
        if conversation:
            self.conversation = conversation
            if system_message:
                self.conversation.system_message = system_message
        else:
            self.conversation = Conversation(
                system_message=system_message,
                max_tokens=self.config.max_tokens,
                reserve_tokens=self.config.reserve_tokens,
            )

        # Request context for tool execution (can be set per-run)
        self._request_context: RequestContext | None = None

    @property
    def model(self) -> type[ModelProfile]:
        """Get the provider's model profile."""
        return self.provider.model

    # === Main API ===

    async def run(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Run the agent to completion.

        The agent will execute tools and continue until:
        - It produces a response without tool calls
        - It reaches the maximum number of turns
        - An error occurs

        Args:
            prompt: User message to start/continue with
            max_turns: Override max turns for this run
            context: Request context for correlation, tracing, and middleware
            **kwargs: Additional arguments passed to provider

        Returns:
            AgentResult with final content and turn history
        """
        max_turns = max_turns or self.config.max_turns

        # Add user message
        self.conversation.add_user(prompt)

        turns: list[TurnResult] = []
        total_usage = Usage()

        for turn_num in range(max_turns):
            # Check for cancellation at the start of each turn
            if context and context.cancellation_token:
                context.cancellation_token.raise_if_cancelled()

            # Get completion
            if self.engine:
                engine_messages = self.conversation.get_messages(model=self.model if self.config.max_tokens else None)
                completion = await self._complete_with_engine(engine_messages, **kwargs)
            else:
                provider_messages = self.conversation.get_messages_dict(
                    model=self.model if self.config.max_tokens else None
                )
                completion = await self.provider.complete(
                    provider_messages,
                    tools=self.tools.tools if self.tools else None,
                    **kwargs,
                )

            # Track usage
            if completion.usage:
                total_usage.input_tokens += completion.usage.input_tokens
                total_usage.output_tokens += completion.usage.output_tokens
                total_usage.total_tokens += completion.usage.total_tokens
                total_usage.total_cost += completion.usage.total_cost

            # Check for errors
            if not completion.ok:
                return AgentResult(
                    content=completion.error,
                    turns=turns,
                    conversation=self.conversation,
                    total_usage=total_usage,
                    status="error",
                    error=completion.error,
                )

            # Create turn result
            turn = TurnResult(
                completion=completion,
                tool_calls=completion.tool_calls or [],
                turn_number=turn_num,
            )

            # Handle tool calls
            if completion.has_tool_calls:
                # Add assistant message with tool calls
                self.conversation.add_assistant_with_tools(
                    completion.content,
                    completion.tool_calls or [],
                )

                # Execute tools with context
                tool_results = await self._execute_tools(
                    completion.tool_calls or [],
                    request_context=context,
                )
                turn.tool_results = tool_results

                # Add tool results to conversation
                for tc, result in zip(completion.tool_calls or [], tool_results, strict=False):
                    self.conversation.add_tool_result(
                        tc.id,
                        result.to_string(),
                        tc.name,
                    )

                # Check for tool errors
                if self.config.stop_on_tool_error:
                    for result in tool_results:
                        if not result.success:
                            turns.append(turn)
                            return AgentResult(
                                content=f"Tool error: {result.error}",
                                turns=turns,
                                conversation=self.conversation,
                                total_usage=total_usage,
                                status="error",
                                error=result.error,
                            )

                turns.append(turn)
                continue  # Next turn

            # No tool calls - we're done
            self.conversation.add_assistant(completion.content or "")
            turns.append(turn)

            return AgentResult(
                content=completion.content,
                turns=turns,
                conversation=self.conversation,
                total_usage=total_usage,
                status="success",
            )

        # Hit max turns
        return AgentResult(
            content=turns[-1].content if turns else None,
            turns=turns,
            conversation=self.conversation,
            total_usage=total_usage,
            status="max_turns",
        )

    async def stream(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the agent's response with tool execution.

        Yields events for:
        - Tokens as they're generated
        - Tool call start/delta/end
        - Tool execution results (as custom events)
        - Usage stats
        - Final result

        Args:
            prompt: User message
            max_turns: Override max turns
            context: Request context for correlation, tracing, and middleware
            **kwargs: Additional provider arguments

        Yields:
            StreamEvent objects
        """
        max_turns = max_turns or self.config.max_turns

        # Add user message
        self.conversation.add_user(prompt)

        total_usage = Usage()
        turns: list[TurnResult] = []

        for turn_num in range(max_turns):
            # Check for cancellation at the start of each turn
            if context and context.cancellation_token:
                context.cancellation_token.raise_if_cancelled()

            # Yield turn start
            yield StreamEvent(type=StreamEventType.META, data={"event": "turn_start", "turn": turn_num})

            # Get messages for this turn
            # Stream completion
            buffer = BufferingAdapter()

            if self.engine:
                engine_messages = self.conversation.get_messages(model=self.model if self.config.max_tokens else None)
                spec = RequestSpec(
                    provider=self.provider.__class__.__name__,
                    model=self.provider.model_name,
                    messages=engine_messages,
                    tools=self.tools.tools if self.tools else None,
                    tool_choice=kwargs.get("tool_choice"),
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    response_format=kwargs.get("response_format"),
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    reasoning=kwargs.get("reasoning"),
                    extra={
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in {
                            "tool_choice",
                            "temperature",
                            "max_tokens",
                            "response_format",
                            "reasoning_effort",
                            "reasoning",
                        }
                    },
                    stream=True,
                )
                stream_iter = self.engine.stream(spec, context=context)
            else:
                provider_messages = self.conversation.get_messages_dict(
                    model=self.model if self.config.max_tokens else None
                )
                stream_iter = self.provider.stream(
                    provider_messages,
                    tools=self.tools.tools if self.tools else None,
                    **kwargs,
                )

            async for event in buffer.wrap(stream_iter):
                # Provider streams typically emit their own DONE event carrying a CompletionResult.
                # Agent.stream() is an agent-level stream and must have a single terminal DONE
                # carrying an AgentResult, so we suppress provider DONE events here.
                if event.type == StreamEventType.DONE:
                    continue
                yield event

            result = buffer.get_result()

            # Track usage
            if result.usage:
                total_usage.input_tokens += result.usage.input_tokens
                total_usage.output_tokens += result.usage.output_tokens
                total_usage.total_tokens += result.usage.total_tokens
                total_usage.total_cost += result.usage.total_cost

            # Create turn result
            turn = TurnResult(
                completion=result,
                tool_calls=result.tool_calls or [],
                turn_number=turn_num,
            )

            # Provider error (stream ended without a successful result)
            if not result.ok:
                turns.append(turn)
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data=AgentResult(
                        content=result.error,
                        turns=turns,
                        conversation=self.conversation,
                        total_usage=total_usage,
                        status="error",
                        error=result.error,
                    ),
                )
                return

            # Handle tool calls
            if result.has_tool_calls:
                # Add assistant message with tool calls
                self.conversation.add_assistant_with_tools(
                    result.content,
                    result.tool_calls or [],
                )

                # Execute tools with context
                tool_results = await self._execute_tools(
                    result.tool_calls or [],
                    request_context=context,
                )
                turn.tool_results = tool_results

                # Yield tool results as events
                for tc, tr in zip(result.tool_calls or [], tool_results, strict=False):
                    yield StreamEvent(
                        type=StreamEventType.META,
                        data={
                            "event": "tool_result",
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "success": tr.success,
                            "content": tr.to_string()[:500],  # Truncate for event
                        },
                    )

                    # Add to conversation
                    self.conversation.add_tool_result(
                        tc.id,
                        tr.to_string(),
                        tc.name,
                    )

                # Optionally stop on tool error (match non-streaming run() behavior)
                if self.config.stop_on_tool_error:
                    for tr in tool_results:
                        if not tr.success:
                            turns.append(turn)
                            yield StreamEvent(
                                type=StreamEventType.DONE,
                                data=AgentResult(
                                    content=f"Tool error: {tr.error}",
                                    turns=turns,
                                    conversation=self.conversation,
                                    total_usage=total_usage,
                                    status="error",
                                    error=tr.error,
                                ),
                            )
                            return

                turns.append(turn)
                continue  # Next turn

            # No tool calls - we're done
            self.conversation.add_assistant(result.content or "")
            turns.append(turn)

            # Yield final result
            yield StreamEvent(
                type=StreamEventType.DONE,
                data=AgentResult(
                    content=result.content,
                    turns=turns,
                    conversation=self.conversation,
                    total_usage=total_usage,
                    status="success",
                ),
            )
            return

        # Hit max turns
        yield StreamEvent(
            type=StreamEventType.DONE,
            data=AgentResult(
                content=turns[-1].content if turns else None,
                turns=turns,
                conversation=self.conversation,
                total_usage=total_usage,
                status="max_turns",
            ),
        )

    async def chat(
        self,
        message: str,
        *,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple chat interface - returns just the response text.

        Args:
            message: User message
            context: Request context for correlation, tracing, and middleware
            **kwargs: Additional arguments

        Returns:
            Assistant's response text
        """
        result = await self.run(message, context=context, **kwargs)
        return result.content or ""

    # === Tool Management ===

    def add_tool(self, tool: Tool) -> Agent:
        """Add a tool to the agent."""
        self.tools.register(tool)
        return self

    def remove_tool(self, name: str) -> Agent:
        """Remove a tool by name."""
        self.tools.unregister(name)
        return self

    async def _execute_tools(
        self,
        tool_calls: list,
        request_context: RequestContext | None = None,
    ) -> list[ToolResult]:
        """Execute tool calls using the execution module with middleware support."""
        limited_calls = tool_calls[: self.config.max_tool_calls_per_turn]
        
        # Use provided context or fall back to agent's context
        ctx = request_context or self._request_context
        
        results = await execute_tools(
            limited_calls,
            self.tools,
            self.config,
            request_context=ctx,
        )
        # Apply output limits
        return [
            apply_tool_output_limit(r, tc.name, self.config.max_tool_output_chars)
            for r, tc in zip(results, limited_calls, strict=False)
        ]

    async def _complete_with_engine(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        engine = self.engine
        if engine is None:
            raise RuntimeError("Engine requested but not configured.")

        cache_response = bool(kwargs.pop("cache_response", False))
        cache_collection = kwargs.pop("cache_collection", None)
        rewrite_cache = bool(kwargs.pop("rewrite_cache", False))
        regen_cache = bool(kwargs.pop("regen_cache", False))

        attempts = kwargs.pop("attempts", None)
        backoff = kwargs.pop("backoff", None)

        tool_choice = kwargs.pop("tool_choice", None)
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)
        response_format = kwargs.pop("response_format", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning = kwargs.pop("reasoning", None)

        retry_cfg = None
        if attempts is not None or backoff is not None:
            retry_cfg = RetryConfig(
                attempts=attempts or 3,
                backoff=backoff or 1.0,
            )

        spec = RequestSpec(
            provider=self.provider.__class__.__name__,
            model=self.provider.model_name,
            messages=messages,
            tools=self.tools.tools if self.tools else None,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            extra=kwargs,
            stream=False,
        )

        return await engine.complete(
            spec,
            cache_response=cache_response,
            cache_collection=cache_collection,
            rewrite_cache=rewrite_cache,
            regen_cache=regen_cache,
            retry=retry_cfg,
        )

    # === Conversation Management ===

    def reset(self) -> Agent:
        """Reset conversation history (preserves system message)."""
        self.conversation.clear()
        return self

    def fork(self) -> Agent:
        """Create a copy of this agent with a forked conversation."""
        return Agent(
            provider=self.provider,
            tools=self.tools,
            conversation=self.conversation.fork(),
            config=self.config,
        )

    # === Session Persistence ===

    def save_session(self, path: str | Path) -> None:
        """
        Save the agent session (conversation + config) to a file.

        This saves:
        - Conversation history (messages, system message)
        - Agent configuration
        - Tool names (tools themselves must be re-registered on load)

        Args:
            path: File path to save the session to (JSON format)
        """
        save_agent_session(path, self.conversation, self.config, self.tools)

    @classmethod
    def load_session(
        cls,
        path: str | Path,
        provider: Provider,
        *,
        tools: list[Tool] | ToolRegistry | None = None,
    ) -> Agent:
        """
        Load an agent session from a file.

        Note: Tools must be provided separately as they cannot be serialized.
        The loaded session will warn if expected tools are missing.

        Args:
            path: File path to load the session from
            provider: LLM provider for the agent
            tools: Tools to register (should include tools used in saved session)

        Returns:
            Agent with restored conversation and configuration
        """
        session_data = load_agent_session(path)

        # Load conversation
        conversation = Conversation.from_dict(session_data["conversation"])

        # Load config
        config_data = session_data.get("config", {})
        config = AgentConfig(
            max_turns=config_data.get("max_turns", 10),
            max_tool_calls_per_turn=config_data.get("max_tool_calls_per_turn", 10),
            parallel_tool_execution=config_data.get("parallel_tool_execution", True),
            tool_timeout=config_data.get("tool_timeout", 30.0),
            max_tool_output_chars=config_data.get("max_tool_output_chars"),
            max_tokens=config_data.get("max_tokens"),
            reserve_tokens=config_data.get("reserve_tokens", 2000),
            stop_on_tool_error=config_data.get("stop_on_tool_error", False),
            include_tool_errors_in_context=config_data.get("include_tool_errors_in_context", True),
            stream_tool_calls=config_data.get("stream_tool_calls", True),
        )

        # Create agent
        agent = cls(provider=provider, tools=tools, conversation=conversation, config=config)

        # Check for missing tools
        expected_tools = set(session_data.get("tool_names", []))
        provided_tools = set(agent.tools.names) if agent.tools else set()
        missing_tools = expected_tools - provided_tools

        if missing_tools:
            warnings.warn(
                f"Session was saved with tools that are not provided: {missing_tools}. "
                "The agent may not work correctly without these tools.",
                UserWarning,
                stacklevel=2,
            )

        return agent

    # === Context Manager ===

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.provider, "close"):
            await self.provider.close()


__all__ = ["Agent"]
