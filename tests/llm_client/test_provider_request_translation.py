from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.google import GoogleProvider
from llm_client.providers.openai import OpenAIProvider
from llm_client.providers.types import CompletionResult, Message, ToolCall
from llm_client.tools import (
    ResponsesBuiltinTool,
    ResponsesConnectorId,
    ResponsesCustomTool,
    ResponsesGrammar,
    ResponsesMCPApprovalPolicy,
    ResponsesMCPTool,
    ResponsesMCPToolFilter,
)
from llm_client.tools.base import Tool
from tests.llm_client.fakes import FakeModel


async def _lookup(q: str) -> str:
    return q


class _LimitContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _NoopLimiter:
    def limit(self, **kwargs):
        _ = kwargs
        return _LimitContext()


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


@pytest.mark.asyncio
async def test_openai_responses_request_translation_omits_non_default_temperature_for_gpt5() -> None:
    provider = _openai_provider("gpt-5")
    provider.use_responses_api = True
    provider.limiter = _NoopLimiter()

    captured: dict[str, object] = {}

    async def _responses_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="resp_controls",
            model="gpt-5",
            status="completed",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="done")])],
            output_text="done",
            usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            incomplete_details=None,
            error=None,
        )

    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=_responses_create),
    )

    result = await provider.complete([Message.user("hi")], temperature=0.0)

    assert result.content == "done"
    assert "temperature" not in captured


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


def test_openai_responses_request_translation_flattens_tools_and_assistant_history() -> None:
    provider = _openai_provider("gpt-5-mini")

    rewritten, alias_to_original, original_to_alias = provider._prepare_openai_tools(
        [_tool()],
        responses_api=True,
    )

    assert rewritten is not None
    assert rewritten[0]["type"] == "function"
    assert rewritten[0]["name"] == "Conversation_Profile_Requirements"
    assert "function" not in rewritten[0]
    assert alias_to_original["Conversation_Profile_Requirements"] == "Conversation.Profile.Requirements"
    assert original_to_alias["Conversation.Profile.Requirements"] == "Conversation_Profile_Requirements"

    payload = provider._messages_to_api_format(
        [
            Message.user("hello"),
            Message.assistant(
                "calling tool",
                tool_calls=[ToolCall(id="call_1", name="Conversation.Profile.Requirements", arguments='{"q":"x"}')],
            ),
            Message.tool_result("call_1", '{"ok":true}', name="Conversation.Profile.Requirements"),
        ],
        responses_api=True,
    )

    assert payload[0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }
    assert payload[1]["type"] == "message"
    assert payload[1]["role"] == "assistant"
    assert payload[1]["content"][0]["type"] == "output_text"
    assert payload[2] == {
        "type": "function_call",
        "id": "call_1",
        "call_id": "call_1",
        "name": "Conversation.Profile.Requirements",
        "arguments": '{"q":"x"}',
        "status": "completed",
    }
    assert payload[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok":true}',
    }


def test_openai_responses_request_translation_preserves_reasoning_items_and_allowed_tools_aliases() -> None:
    provider = _openai_provider("gpt-5-mini")

    preserved_result = CompletionResult(
        content="calling tool",
        tool_calls=[ToolCall(id="call_1", name="Conversation.Profile.Requirements", arguments='{"q":"x"}')],
        reasoning="thought",
        provider_items=[
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "content": [],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "Conversation_Profile_Requirements",
                "arguments": '{"q":"x"}',
                "status": "completed",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "calling tool"}],
            },
        ],
    )

    payload = provider._messages_to_api_format(
        [
            Message.user("hello"),
            preserved_result.to_message(),
            Message.tool_result("call_1", '{"ok":true}', name="Conversation.Profile.Requirements"),
        ],
        responses_api=True,
    )

    assert payload[1]["type"] == "reasoning"
    assert payload[2] == {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "Conversation_Profile_Requirements",
        "arguments": '{"q":"x"}',
        "status": "completed",
    }
    assert payload[3]["type"] == "message"
    assert payload[4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok":true}',
    }

    _, _, original_to_alias = provider._prepare_openai_tools([_tool()], responses_api=True)
    normalized_choice = provider._normalize_tool_choice(
        {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "function", "name": "Conversation.Profile.Requirements"},
            ],
        },
        use_responses_api=True,
        original_to_alias=original_to_alias,
    )

    assert normalized_choice == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "name": "Conversation_Profile_Requirements"},
        ],
    }


def test_openai_responses_request_translation_supports_builtin_and_custom_tool_descriptors() -> None:
    provider = _openai_provider("gpt-5-mini")
    strict_tool = Tool(
        name="Lookup.Strict",
        description="Lookup strict data",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=_lookup,
        strict=True,
    )

    rewritten, alias_to_original, original_to_alias = provider._prepare_openai_tools(
        [
            ResponsesBuiltinTool.file_search(vector_store_ids=["vs_123"], max_num_results=5),
            ResponsesCustomTool(
                name="planner",
                description="Emit a terse plan.",
                grammar=ResponsesGrammar(syntax="lark", definition='start: "done"'),
            ),
            strict_tool,
        ],
        responses_api=True,
    )

    assert rewritten is not None
    assert rewritten[0] == {
        "type": "file_search",
        "vector_store_ids": ["vs_123"],
        "max_num_results": 5,
    }
    assert rewritten[1] == {
        "type": "custom",
        "name": "planner",
        "description": "Emit a terse plan.",
        "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "done"'},
    }
    assert rewritten[2]["type"] == "function"
    assert rewritten[2]["name"] == "Lookup_Strict"
    assert rewritten[2]["strict"] is True
    assert alias_to_original["Lookup_Strict"] == "Lookup.Strict"
    assert original_to_alias["Lookup.Strict"] == "Lookup_Strict"


def test_openai_responses_request_translation_defaults_function_tools_to_strict() -> None:
    provider = _openai_provider("gpt-5-mini")

    rewritten, _, _ = provider._prepare_openai_tools(
        [_tool("Lookup.DefaultStrict")],
        responses_api=True,
    )

    assert rewritten is not None
    assert rewritten[0]["type"] == "function"
    assert rewritten[0]["name"] == "Lookup_DefaultStrict"
    assert rewritten[0]["strict"] is True


def test_openai_responses_request_translation_supports_typed_mcp_tools_and_policies() -> None:
    provider = _openai_provider("gpt-5-mini")
    policy = ResponsesMCPApprovalPolicy(
        never=ResponsesMCPToolFilter(tool_names=("read_wiki_structure", "ask_question")),
    )

    rewritten, _, _ = provider._prepare_openai_tools(
        [
            ResponsesMCPTool.remote_server(
                "https://mcp.example.com",
                server_label="Research Wiki",
                authorization="Bearer token",
                allowed_tools=["read_wiki_structure", "ask_question"],
                require_approval=policy,
            )
        ],
        responses_api=True,
    )

    assert rewritten == [
        {
            "type": "mcp",
            "server_label": "Research Wiki",
            "server_url": "https://mcp.example.com",
            "authorization": "Bearer token",
            "allowed_tools": ["read_wiki_structure", "ask_question"],
            "require_approval": {
                "never": {"tool_names": ["read_wiki_structure", "ask_question"]},
            },
        }
    ]

    connector = ResponsesMCPTool.connector(
        ResponsesConnectorId.GMAIL,
        authorization="Bearer oauth-token",
    )
    connector_payload = connector.to_dict()
    assert connector_payload["connector_id"] == "connector_gmail"
    assert connector_payload["authorization"] == "Bearer oauth-token"


@pytest.mark.asyncio
async def test_openai_start_deep_research_normalizes_typed_mcp_tools() -> None:
    provider = _openai_provider("o3-deep-research")
    provider.use_responses_api = True
    captured: list[dict[str, object]] = []

    async def _complete(messages, **kwargs):
        captured.append({"messages": messages, **kwargs})
        return SimpleNamespace(ok=True, content="queued")

    provider.complete = _complete  # type: ignore[method-assign]

    await provider.start_deep_research(
        "Research semaglutide",
        web_search=True,
        mcp_tools=[
            ResponsesMCPTool.deep_research_remote_server(
                "https://mcp.example.com",
                server_label="Research Wiki",
                allowed_tools=["read_wiki_structure"],
            )
        ],
    )

    rendered_tools = [
        tool.to_dict() if hasattr(tool, "to_dict") else tool
        for tool in captured[0]["tools"]
    ]

    assert rendered_tools == [
        {"type": "web_search_preview"},
        {
            "type": "mcp",
            "server_label": "Research Wiki",
            "server_url": "https://mcp.example.com",
            "allowed_tools": ["read_wiki_structure"],
            "require_approval": "never",
        },
    ]


def test_openai_responses_native_tool_descriptors_require_responses_api() -> None:
    provider = _openai_provider("gpt-5-mini")

    with pytest.raises(ValueError, match="use_responses_api=True"):
        provider._validate_tool_configuration(
            tools=[ResponsesBuiltinTool.web_search(search_context_size="medium")],
            use_responses_api=False,
        )


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
