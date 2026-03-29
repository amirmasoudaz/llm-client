from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from llm_client.agent import (
    Agent,
    AgentDefinition,
    AgentExecutionPolicy,
    AgentMemoryPolicy,
    AgentOutputPolicy,
    PromptTemplateReference,
    ToolExecutionMode,
)
from llm_client.content import ContentRequestEnvelope, ContentResponseEnvelope
from llm_client.conversation import Conversation
from llm_client.providers.types import CompletionResult, Usage


class _FakeProvider:
    model_name = "gpt-5-mini"
    model = SimpleNamespace(key="gpt-5-mini", count_tokens=lambda content: len(str(content or "").split()))


class _FakeEngine:
    def __init__(self) -> None:
        self.completed: list[ContentRequestEnvelope] = []

    async def complete_content(self, request, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.completed.append(request)
        return ContentResponseEnvelope.from_completion_result(
            CompletionResult(
                content="defined agent completion",
                usage=Usage(total_tokens=4),
                model=request.model,
                status=200,
            )
        )


def test_agent_definition_round_trips_with_policy_objects() -> None:
    definition = AgentDefinition(
        name="research-assistant",
        system_message="You are precise.",
        prompt_templates=(
            PromptTemplateReference(
                name="system-template",
                uri="pkg://prompts/system.md",
                variables=("topic",),
            ),
        ),
        execution_policy=AgentExecutionPolicy(
            max_turns=7,
            max_tool_calls_per_turn=3,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
            tool_timeout=12.0,
            tool_retry_attempts=2,
            batch_concurrency=5,
            stop_on_tool_error=True,
        ),
        output_policy=AgentOutputPolicy(max_tool_output_chars=250),
        memory_policy=AgentMemoryPolicy(max_tokens=2048, reserve_tokens=256, summarization_enabled=True),
        metadata={"owner": "tests"},
    )

    config = definition.to_agent_config()
    rebuilt = AgentDefinition.from_agent_config(
        config,
        name=definition.name,
        system_message=definition.system_message,
        prompt_templates=definition.prompt_templates,
        metadata=definition.metadata,
    )

    assert config.max_turns == 7
    assert config.parallel_tool_execution is False
    assert config.max_tool_output_chars == 250
    assert config.max_tokens == 2048
    assert rebuilt.execution_policy.tool_execution_mode is ToolExecutionMode.SEQUENTIAL
    assert rebuilt.prompt_templates[0].uri == "pkg://prompts/system.md"
    assert rebuilt.metadata["owner"] == "tests"


def test_agent_definition_is_immutable() -> None:
    definition = AgentDefinition(system_message="immutable")

    with pytest.raises(FrozenInstanceError):
        definition.system_message = "mutated"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_agent_uses_definition_and_keeps_runtime_state_separate() -> None:
    engine = _FakeEngine()
    definition = AgentDefinition(
        name="defined-agent",
        system_message="You are precise.",
        prompt_templates=(PromptTemplateReference(name="inline", inline_text="Prompt body"),),
        execution_policy=AgentExecutionPolicy(max_turns=4),
        output_policy=AgentOutputPolicy(max_tool_output_chars=120),
        memory_policy=AgentMemoryPolicy(max_tokens=1024, reserve_tokens=128),
        metadata={"team": "core"},
    )
    conversation = Conversation(system_message="Conversation system")
    agent = Agent(provider=_FakeProvider(), engine=engine, definition=definition, conversation=conversation)

    result = await agent.run("hello")

    assert result.status == "success"
    assert agent.definition.name == "defined-agent"
    assert agent.definition.system_message == "You are precise."
    assert agent.definition.prompt_templates[0].inline_text == "Prompt body"
    assert agent.definition.output_policy.max_tool_output_chars == 120
    assert agent.definition.memory_policy.max_tokens == 1024
    assert agent.conversation.system_message == "You are precise."
    assert agent._runtime.request_context is None
    assert engine.completed[0].model == "gpt-5-mini"


def test_agent_definition_inherits_conversation_system_message_when_needed() -> None:
    conversation = Conversation(system_message="Conversation system")
    agent = Agent(provider=_FakeProvider(), conversation=conversation)

    assert agent.definition.system_message == "Conversation system"
