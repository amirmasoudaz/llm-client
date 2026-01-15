"""
Agent orchestration with ReAct loop and tool calling.

This module provides the Agent class that composes providers, tools,
and conversations into an autonomous agent capable of multi-turn
reasoning and action.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from .conversation import Conversation
from .providers.base import Provider
from .providers.types import (
    CompletionResult,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from .streaming import BufferingAdapter
from .tools.base import Tool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from .models import ModelProfile


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # Turn limits
    max_turns: int = 10
    max_tool_calls_per_turn: int = 10
    
    # Tool execution
    parallel_tool_execution: bool = True
    tool_timeout: float = 30.0
    
    # Context management
    max_tokens: Optional[int] = None
    reserve_tokens: int = 2000
    
    # Behavior flags
    stop_on_tool_error: bool = False
    include_tool_errors_in_context: bool = True
    
    # Streaming
    stream_tool_calls: bool = True


@dataclass
class TurnResult:
    """Result of a single agent turn."""
    
    completion: CompletionResult
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    turn_number: int = 0
    
    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
    
    @property
    def content(self) -> Optional[str]:
        return self.completion.content


@dataclass
class AgentResult:
    """Final result of an agent run."""
    
    content: Optional[str] = None
    turns: List[TurnResult] = field(default_factory=list)
    conversation: Optional[Conversation] = None
    
    # Aggregated usage
    total_usage: Optional[Usage] = None
    
    # Status
    status: Literal["success", "max_turns", "error"] = "success"
    error: Optional[str] = None
    
    @property
    def num_turns(self) -> int:
        return len(self.turns)
    
    @property
    def all_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls across all turns."""
        calls = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "num_turns": self.num_turns,
            "status": self.status,
            "error": self.error,
            "total_usage": self.total_usage.to_dict() if self.total_usage else None,
        }


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
        tools: Optional[Union[List[Tool], ToolRegistry]] = None,
        system_message: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        config: Optional[AgentConfig] = None,
        # Shortcuts for common config options
        max_turns: int = 10,
        max_tokens: Optional[int] = None,
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
        """
        self.provider = provider
        
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
    
    @property
    def model(self) -> Type["ModelProfile"]:
        """Get the provider's model profile."""
        return self.provider.model
    
    # === Main API ===
    
    async def run(
        self,
        prompt: str,
        *,
        max_turns: Optional[int] = None,
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
            **kwargs: Additional arguments passed to provider
            
        Returns:
            AgentResult with final content and turn history
        """
        max_turns = max_turns or self.config.max_turns
        
        # Add user message
        self.conversation.add_user(prompt)
        
        turns: List[TurnResult] = []
        total_usage = Usage()
        
        for turn_num in range(max_turns):
            # Get completion
            messages = self.conversation.get_messages_dict(
                model=self.model if self.config.max_tokens else None
            )
            
            completion = await self.provider.complete(
                messages,
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
                
                # Execute tools
                tool_results = await self._execute_tools(completion.tool_calls or [])
                turn.tool_results = tool_results
                
                # Add tool results to conversation
                for tc, result in zip(completion.tool_calls or [], tool_results):
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
        max_turns: Optional[int] = None,
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
            **kwargs: Additional provider arguments
            
        Yields:
            StreamEvent objects
        """
        import time
        
        max_turns = max_turns or self.config.max_turns
        
        # Add user message
        self.conversation.add_user(prompt)
        
        total_usage = Usage()
        turns: List[TurnResult] = []
        
        for turn_num in range(max_turns):
            # Yield turn start
            yield StreamEvent(
                type=StreamEventType.META,
                data={"event": "turn_start", "turn": turn_num}
            )
            
            # Get messages for this turn
            messages = self.conversation.get_messages_dict(
                model=self.model if self.config.max_tokens else None
            )
            
            # Stream completion
            buffer = BufferingAdapter()
            
            async for event in buffer.wrap(
                self.provider.stream(
                    messages,
                    tools=self.tools.tools if self.tools else None,
                    **kwargs,
                )
            ):
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
            
            # Handle tool calls
            if result.has_tool_calls:
                # Add assistant message with tool calls
                self.conversation.add_assistant_with_tools(
                    result.content,
                    result.tool_calls or [],
                )
                
                # Execute tools
                tool_results = await self._execute_tools(result.tool_calls or [])
                turn.tool_results = tool_results
                
                # Yield tool results as events
                for tc, tr in zip(result.tool_calls or [], tool_results):
                    yield StreamEvent(
                        type=StreamEventType.META,
                        data={
                            "event": "tool_result",
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "success": tr.success,
                            "content": tr.to_string()[:500],  # Truncate for event
                        }
                    )
                    
                    # Add to conversation
                    self.conversation.add_tool_result(
                        tc.id,
                        tr.to_string(),
                        tc.name,
                    )
                
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
                )
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
            )
        )
    
    async def chat(self, message: str, **kwargs: Any) -> str:
        """
        Simple chat interface - returns just the response text.
        
        Args:
            message: User message
            **kwargs: Additional arguments
            
        Returns:
            Assistant's response text
        """
        result = await self.run(message, **kwargs)
        return result.content or ""
    
    # === Tool Management ===
    
    def add_tool(self, tool: Tool) -> "Agent":
        """Add a tool to the agent."""
        self.tools.register(tool)
        return self
    
    def remove_tool(self, name: str) -> "Agent":
        """Remove a tool by name."""
        self.tools.unregister(name)
        return self
    
    async def _execute_tools(
        self,
        tool_calls: List[ToolCall],
    ) -> List[ToolResult]:
        """Execute tool calls, optionally in parallel."""
        if not tool_calls:
            return []
        
        # Limit tool calls per turn
        tool_calls = tool_calls[:self.config.max_tool_calls_per_turn]
        
        if self.config.parallel_tool_execution:
            # Execute all tools in parallel
            tasks = [
                self._execute_single_tool(tc)
                for tc in tool_calls
            ]
            return await asyncio.gather(*tasks)
        else:
            # Execute sequentially
            results = []
            for tc in tool_calls:
                result = await self._execute_single_tool(tc)
                results.append(result)
            return results
    
    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with timeout."""
        try:
            result = await asyncio.wait_for(
                self.tools.execute(tool_call.name, tool_call.arguments),
                timeout=self.config.tool_timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ToolResult.error_result(
                f"Tool '{tool_call.name}' timed out after {self.config.tool_timeout}s"
            )
        except Exception as e:
            return ToolResult.error_result(f"Tool execution error: {e}")
    
    # === Conversation Management ===
    
    def reset(self) -> "Agent":
        """Reset conversation history (preserves system message)."""
        self.conversation.clear()
        return self
    
    def fork(self) -> "Agent":
        """Create a copy of this agent with a forked conversation."""
        return Agent(
            provider=self.provider,
            tools=self.tools,
            conversation=self.conversation.fork(),
            config=self.config,
        )
    
    # === Session Persistence ===
    
    def save_session(self, path: Union[str, Path]) -> None:
        """
        Save the agent session (conversation + config) to a file.
        
        This saves:
        - Conversation history (messages, system message)
        - Agent configuration
        - Tool names (tools themselves must be re-registered on load)
        
        Args:
            path: File path to save the session to (JSON format)
            
        Example:
            ```python
            agent = Agent(provider=provider, system_message="You are helpful.")
            await agent.run("Hello!")
            agent.save_session("session.json")
            
            # Later...
            loaded_agent = Agent.load_session("session.json", provider=provider)
            ```
        """
        path = Path(path)
        
        session_data = {
            "version": "1.0",
            "conversation": self.conversation.to_dict(),
            "config": {
                "max_turns": self.config.max_turns,
                "max_tool_calls_per_turn": self.config.max_tool_calls_per_turn,
                "parallel_tool_execution": self.config.parallel_tool_execution,
                "tool_timeout": self.config.tool_timeout,
                "max_tokens": self.config.max_tokens,
                "reserve_tokens": self.config.reserve_tokens,
                "stop_on_tool_error": self.config.stop_on_tool_error,
                "include_tool_errors_in_context": self.config.include_tool_errors_in_context,
                "stream_tool_calls": self.config.stream_tool_calls,
            },
            "tool_names": self.tools.names if self.tools else [],
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session_data, indent=2))
    
    @classmethod
    def load_session(
        cls,
        path: Union[str, Path],
        provider: Provider,
        *,
        tools: Optional[Union[List[Tool], ToolRegistry]] = None,
    ) -> "Agent":
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
            
        Example:
            ```python
            # Define the same tools as the original session
            @tool
            async def search(query: str) -> str:
                return f"Results for {query}"
            
            agent = Agent.load_session(
                "session.json",
                provider=OpenAIProvider(model="gpt-5"),
                tools=[search]
            )
            
            # Continue the conversation
            result = await agent.run("What else can you tell me?")
            ```
        """
        path = Path(path)
        session_data = json.loads(path.read_text())
        
        # Load conversation
        conversation = Conversation.from_dict(session_data["conversation"])
        
        # Load config
        config_data = session_data.get("config", {})
        config = AgentConfig(
            max_turns=config_data.get("max_turns", 10),
            max_tool_calls_per_turn=config_data.get("max_tool_calls_per_turn", 10),
            parallel_tool_execution=config_data.get("parallel_tool_execution", True),
            tool_timeout=config_data.get("tool_timeout", 30.0),
            max_tokens=config_data.get("max_tokens"),
            reserve_tokens=config_data.get("reserve_tokens", 2000),
            stop_on_tool_error=config_data.get("stop_on_tool_error", False),
            include_tool_errors_in_context=config_data.get("include_tool_errors_in_context", True),
            stream_tool_calls=config_data.get("stream_tool_calls", True),
        )
        
        # Create agent
        agent = cls(
            provider=provider,
            tools=tools,
            conversation=conversation,
            config=config
        )
        
        # Check for missing tools
        expected_tools = set(session_data.get("tool_names", []))
        provided_tools = set(agent.tools.names) if agent.tools else set()
        missing_tools = expected_tools - provided_tools
        
        if missing_tools:
            import warnings
            warnings.warn(
                f"Session was saved with tools that are not provided: {missing_tools}. "
                "The agent may not work correctly without these tools.",
                UserWarning,
            )
        
        return agent
    
    # === Context Manager ===
    
    async def __aenter__(self) -> "Agent":
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.provider, "close"):
            await self.provider.close()


# === Utility Functions ===

async def quick_agent(
    prompt: str,
    *,
    model: str = "gpt-5-mini",
    tools: Optional[List[Tool]] = None,
    system_message: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Quick one-shot agent call.
    
    Convenience function for simple agent interactions without
    explicitly creating provider and agent instances.
    
    Args:
        prompt: User message
        model: Model key
        tools: Optional tools
        system_message: Optional system message
        **kwargs: Additional arguments
        
    Returns:
        Agent's response text
        
    Example:
        ```python
        response = await quick_agent(
            "What's 2 + 2?",
            model="gpt-5-nano",
        )
        ```
    """
    from .providers.openai import OpenAIProvider
    
    async with OpenAIProvider(model=model) as provider:
        agent = Agent(
            provider=provider,
            tools=tools,
            system_message=system_message,
        )
        result = await agent.run(prompt, **kwargs)
        return result.content or ""


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "TurnResult",
    "quick_agent",
]

