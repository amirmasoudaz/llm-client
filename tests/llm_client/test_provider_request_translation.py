from __future__ import annotations

from types import SimpleNamespace

from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.google import GoogleProvider
from llm_client.providers.openai import OpenAIProvider
from llm_client.providers.types import Message, ToolCall
from llm_client.tools.base import Tool
from tests.llm_client.fakes import FakeModel


async def _lookup(q: str) -> str:
    return q


def _tool(name: str = "Conversation.Profile.Requirements") -> Tool:
    return Tool(
        name=name,
        description="Lookup requirements",
        parameters={
            "oneOf": [{"type": "object"}],
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        handler=_lookup,
    )


def _openai_provider(model_name: str = "gpt-5-mini") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._model = FakeModel(key=model_name, model_name=model_name)
    return provider


def test_openai_request_translation_handles_gpt5_token_and_temperature_rules() -> None:
    provider = _openai_provider("gpt-5-mini")

    params: dict[str, object] = {}
    provider._set_temperature(params, 0.2)
    provider._set_completion_token_limit(params, 123)

    assert "temperature" not in params
    assert params["max_completion_tokens"] == 123

    params = {}
    provider._set_temperature(params, 1.0)
    assert params["temperature"] == 1.0

    provider = _openai_provider("gpt-4o-mini")
    params = {}
    provider._set_temperature(params, 0.2)
    provider._set_completion_token_limit(params, 55)
    assert params["temperature"] == 0.2
    assert params["max_tokens"] == 55


def test_openai_request_translation_sanitizes_tool_names_and_schemas() -> None:
    provider = _openai_provider("gpt-5-mini")

    rewritten, alias_to_original, original_to_alias = provider._prepare_openai_tools([_tool(), _tool("a b c"), _tool("a/b/c")])

    assert rewritten is not None
    aliases = [item["function"]["name"] for item in rewritten]
    assert aliases[0] == "Conversation_Profile_Requirements"
    assert aliases[1] == "a_b_c"
    assert aliases[2] == "a_b_c_2"
    assert alias_to_original["Conversation_Profile_Requirements"] == "Conversation.Profile.Requirements"
    assert original_to_alias["a/b/c"] == "a_b_c_2"
    assert "oneOf" not in rewritten[0]["function"]["parameters"]


def test_google_request_translation_preserves_system_and_tool_results() -> None:
    system_instruction, history = GoogleProvider._convert_messages(
        [
            Message.system("You are helpful."),
            Message.user("hello"),
            Message.assistant(content="Calling tool", tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')]),
            Message.tool_result("call_1", '{"result":"ok"}', name="lookup"),
        ]
    )

    assert system_instruction == "You are helpful."
    assert len(history) == 3
    assert any(getattr(part, "function_call", None) and part.function_call.name == "lookup" for part in history[1].parts)
    assert any(getattr(part, "function_response", None) and part.function_response.name == "lookup" for part in history[2].parts)


def test_anthropic_request_translation_preserves_system_and_tool_results() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._model = FakeModel(key="claude-4-5-sonnet", model_name="claude-4-5-sonnet")

    system_message, messages = provider._convert_messages_for_anthropic(
        [
            Message.system("You are helpful."),
            Message.user("hello"),
            Message.assistant(content="Calling tool", tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')]),
            Message.tool_result("call_1", '{"result":"ok"}', name="lookup"),
        ]
    )

    assert system_message == "You are helpful."
    assert len(messages) == 3
    assert any(block["type"] == "tool_use" for block in messages[1]["content"])
    assert messages[2]["content"][0]["type"] == "tool_result"
