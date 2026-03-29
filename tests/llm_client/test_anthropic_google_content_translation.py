from __future__ import annotations

from types import SimpleNamespace

from llm_client.content import AudioBlock, FileBlock, ImageBlock, TextBlock, ToolResultBlock
from llm_client.providers.anthropic import AnthropicProvider
from llm_client.providers.google import GoogleProvider
from llm_client.providers.types import Message, Role


def test_anthropic_provider_converts_canonical_blocks_into_message_content() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    system_message, messages = AnthropicProvider._convert_messages_for_anthropic(
        provider,
        [
            Message.system([TextBlock("system prompt")]),
            Message.user([TextBlock("hello"), ImageBlock("https://example.com/x.png")]),
            Message.tool_result("call_1", [ToolResultBlock(tool_call_id="call_1", content="done", name="lookup")], name="lookup"),
        ],
    )

    assert system_message == "system prompt"
    assert messages[0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "[image] https://example.com/x.png"},
        ],
    }
    assert messages[1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "done",
            }
        ],
    }


class _FakePart:
    @staticmethod
    def from_text(*, text: str) -> dict[str, str]:
        return {"type": "text", "text": text}

    @staticmethod
    def from_function_response(*, name: str, response: object) -> dict[str, object]:
        return {"type": "function_response", "name": name, "response": response}

    @staticmethod
    def from_function_call(*, name: str, args: object) -> dict[str, object]:
        return {"type": "function_call", "name": name, "args": args}


class _FakeContent:
    def __init__(self, *, role: str, parts: list[object]) -> None:
        self.role = role
        self.parts = parts


def test_google_provider_converts_canonical_blocks_into_parts(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_client.providers.google.types",
        SimpleNamespace(Part=_FakePart, Content=_FakeContent),
    )

    system_instruction, history = GoogleProvider._convert_messages(
        [
            Message.system([TextBlock("system prompt")]),
            Message.user([TextBlock("hello"), ImageBlock("https://example.com/x.png"), AudioBlock(audio_url="https://example.com/a.mp3")]),
            Message.tool_result("call_1", [ToolResultBlock(tool_call_id="call_1", content='{"ok": true}', name="lookup")], name="lookup"),
            Message.user([FileBlock(file_url="https://example.com/doc.pdf")]),
        ]
    )

    assert system_instruction == "system prompt"
    assert history[0].role == "user"
    assert history[0].parts == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "[image] https://example.com/x.png"},
        {"type": "text", "text": "[audio] https://example.com/a.mp3"},
    ]
    assert history[1].parts == [
        {"type": "function_response", "name": "lookup", "response": {"ok": True}},
    ]
    assert history[2].parts == [
        {"type": "text", "text": "[file] https://example.com/doc.pdf"},
    ]
