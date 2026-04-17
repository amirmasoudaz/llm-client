from llm_client.content import Message
from llm_client.request_builders import build_request_spec, infer_model_name, infer_provider_name
from llm_client.tools import (
    ResponsesBuiltinTool,
    ResponsesConnectorId,
    ResponsesCustomTool,
    ResponsesFunctionTool,
    ResponsesGrammar,
    ResponsesMCPApprovalPolicy,
    ResponsesMCPTool,
    ResponsesMCPToolFilter,
    ResponsesToolNamespace,
    ResponsesToolSearch,
)


class _FakeModel:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeProvider:
    def __init__(self) -> None:
        self.model = _FakeModel("fake-model")


def test_request_builder_infers_provider_model_and_extra_fields() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs={
            "temperature": 0.2,
            "max_tokens": 64,
            "response_format": "json_object",
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "tenant-a",
            "prompt_cache_retention": "24h",
            "custom_flag": True,
        },
    )

    assert spec.provider == "fake"
    assert spec.model == "fake-model"
    assert spec.temperature == 0.2
    assert spec.max_tokens == 64
    assert spec.response_format == "json_object"
    assert spec.include == ["reasoning.encrypted_content"]
    assert spec.prompt_cache_key == "tenant-a"
    assert spec.prompt_cache_retention == "24h"
    assert spec.extra == {"custom_flag": True}
    assert spec.messages == [Message.user("hi")]


def test_request_builder_prefers_explicit_model_override() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        model="override-model",
        request_kwargs={"model": "ignored-request-model"},
    )

    assert spec.model == "override-model"
    assert infer_provider_name(provider) == "fake"
    assert infer_model_name(provider) == "fake-model"


def test_request_builder_serializes_provider_format_tool_dicts() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        request_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_profile",
                        "description": "Lookup a profile",
                        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    },
                }
            ]
        },
    )

    payload = spec.to_dict()
    assert isinstance(payload["tools"], list)
    assert payload["tools"][0]["name"] == "lookup_profile"
    assert payload["tools"][0]["provider_definition"]["type"] == "function"


def test_request_builder_serializes_responses_tool_descriptors() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        request_kwargs={
            "tools": [
                ResponsesBuiltinTool.file_search(vector_store_ids=["vs_123"]),
                ResponsesCustomTool(
                    name="planner",
                    description="Emit a compact plan.",
                    grammar=ResponsesGrammar(syntax="lark", definition='start: "done"'),
                ),
            ]
        },
    )

    payload = spec.to_dict()
    assert payload["tools"][0]["name"] == "file_search"
    assert payload["tools"][0]["provider_definition"] == {
        "type": "file_search",
        "vector_store_ids": ["vs_123"],
    }
    assert payload["tools"][1]["name"] == "planner"
    assert payload["tools"][1]["provider_definition"] == {
        "type": "custom",
        "name": "planner",
        "description": "Emit a compact plan.",
        "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "done"'},
    }


def test_request_builder_serializes_advanced_openai_tool_search_and_namespaces() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        request_kwargs={
            "tools": [
                ResponsesToolSearch.client(parameters={"type": "object", "properties": {"query": {"type": "string"}}}),
                ResponsesToolNamespace(
                    name="crm",
                    description="CRM tools",
                    tools=(
                        ResponsesFunctionTool(
                            name="lookup_profile",
                            description="Lookup a profile.",
                            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
                            defer_loading=True,
                        ),
                    ),
                ),
            ]
        },
    )

    payload = spec.to_dict()
    tools_by_name = {tool["name"]: tool for tool in payload["tools"]}
    assert tools_by_name["tool_search"]["provider_definition"] == {
        "type": "tool_search",
        "execution": "client",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    assert tools_by_name["crm"]["provider_definition"] == {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [
            {
                "type": "function",
                "name": "lookup_profile",
                "description": "Lookup a profile.",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                "defer_loading": True,
            }
        ],
    }


def test_responses_builtin_tool_aliases_preserve_provider_shape() -> None:
    connector = ResponsesBuiltinTool.connector(server_label="gmail", require_approval="always")
    remote_mcp = ResponsesBuiltinTool.remote_mcp(server_url="https://mcp.example.com", require_approval="never")
    web_search_preview = ResponsesBuiltinTool.web_search_preview(search_context_size="low")

    assert connector.to_dict() == {
        "type": "mcp",
        "server_label": "gmail",
        "require_approval": "always",
    }
    assert remote_mcp.to_dict() == {
        "type": "mcp",
        "server_url": "https://mcp.example.com",
        "require_approval": "never",
    }
    assert web_search_preview.to_dict() == {
        "type": "web_search_preview",
        "search_context_size": "low",
    }


def test_request_builder_serializes_typed_mcp_tools() -> None:
    provider = _FakeProvider()
    policy = ResponsesMCPApprovalPolicy(
        never=ResponsesMCPToolFilter(tool_names=("read_wiki_structure", "ask_question")),
    )

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        request_kwargs={
            "tools": [
                    ResponsesMCPTool.remote_server(
                        "https://mcp.example.com",
                        server_label="Research Wiki",
                        authorization="Bearer token",
                        allowed_tools=["read_wiki_structure", "ask_question"],
                        require_approval=policy,
                    )
            ]
        },
    )

    payload = spec.to_dict()
    assert payload["tools"][0]["name"] == "Research Wiki"
    assert payload["tools"][0]["provider_definition"] == {
        "type": "mcp",
        "server_label": "Research Wiki",
        "server_url": "https://mcp.example.com",
        "authorization": "Bearer token",
        "allowed_tools": ["read_wiki_structure", "ask_question"],
        "require_approval": {
            "never": {
                "tool_names": ["read_wiki_structure", "ask_question"],
            }
        },
    }


def test_typed_connector_enum_serializes_to_documented_connector_id() -> None:
    tool = ResponsesMCPTool.connector(
        ResponsesConnectorId.GOOGLE_CALENDAR,
        authorization="Bearer oauth-token",
    )

    assert tool.to_dict() == {
        "type": "mcp",
        "connector_id": "connector_googlecalendar",
        "authorization": "Bearer oauth-token",
    }
