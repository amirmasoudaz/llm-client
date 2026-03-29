from llm_client.content import (
    AudioBlock,
    ContentMessage,
    ContentRequestEnvelope,
    ContentResponseEnvelope,
    ContentBlockType,
    ContentHandlingMode,
    FileBlock,
    ImageBlock,
    MetadataBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    content_block_from_dict,
    content_blocks_to_text,
    project_content_blocks,
    message_from_content_blocks,
    message_to_content_blocks,
    normalize_content_blocks,
    serialize_message_content,
    UnsupportedContentError,
)
from llm_client.providers.types import CompletionResult, Message, Role, Usage


def test_content_blocks_round_trip_from_dict_and_text_projection() -> None:
    blocks = normalize_content_blocks(
        [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "text": "plan"},
            {"type": "tool_result", "tool_call_id": "call_1", "content": "done"},
        ]
    )

    assert [block.type for block in blocks] == [
        ContentBlockType.TEXT,
        ContentBlockType.REASONING,
        ContentBlockType.TOOL_RESULT,
    ]
    assert content_blocks_to_text(blocks) == "hello\nplan\ndone"
    assert isinstance(content_block_from_dict({"type": "image", "image_url": "https://example.com/x.png"}), ImageBlock)


def test_message_serialization_accepts_block_objects() -> None:
    message = Message.user(
        [
            TextBlock("hello"),
            ImageBlock("https://example.com/x.png", detail="high"),
            AudioBlock(audio_url="https://example.com/a.mp3", mime_type="audio/mpeg"),
            FileBlock(file_url="https://example.com/doc.pdf", name="doc.pdf"),
            MetadataBlock(data={"source": "test"}),
        ]
    )

    payload = message.to_dict()

    assert isinstance(payload["content"], list)
    assert payload["content"][0]["text"] == "hello"
    assert payload["content"][1]["image_url"] == "https://example.com/x.png"
    assert payload["content"][2]["audio_url"] == "https://example.com/a.mp3"
    assert payload["content"][3]["file_url"] == "https://example.com/doc.pdf"
    assert payload["content"][4]["data"] == {"source": "test"}


def test_message_block_adapters_bridge_tool_calls_and_results() -> None:
    blocks = [
        TextBlock("hello"),
        ToolCallBlock(id="call_1", name="lookup", arguments='{"q":"hello"}'),
    ]
    message = message_from_content_blocks(role=Role.ASSISTANT, blocks=blocks)
    round_tripped = message_to_content_blocks(message)

    assert message.tool_calls is not None
    assert message.tool_calls[0].name == "lookup"
    assert isinstance(message.content, list)
    assert isinstance(round_tripped[0], TextBlock)
    assert isinstance(round_tripped[1], ToolCallBlock)

    tool_message = message_from_content_blocks(
        role=Role.TOOL,
        blocks=[ToolResultBlock(tool_call_id="call_1", content="done", name="lookup")],
        name="lookup",
    )
    tool_blocks = message_to_content_blocks(tool_message)

    assert tool_message.tool_call_id == "call_1"
    assert tool_message.content == "done"
    assert len(tool_blocks) == 1
    assert isinstance(tool_blocks[0], ToolResultBlock)


def test_serialize_message_content_preserves_plain_text_and_block_payloads() -> None:
    assert serialize_message_content("hello") == "hello"
    payload = serialize_message_content([TextBlock("hi"), ReasoningBlock("think")])

    assert payload == [
        {"type": "text", "text": "hi"},
        {"type": "reasoning", "text": "think"},
    ]


def test_content_request_envelope_round_trips_request_spec() -> None:
    envelope = ContentRequestEnvelope(
        provider="openai",
        model="gpt-5",
        messages=(
            ContentMessage(role=Role.USER, blocks=(TextBlock("hello"), ImageBlock("https://example.com/x.png"))),
        ),
        temperature=0.2,
        max_tokens=128,
        response_format="json_object",
        extra={"trace": "yes"},
        stream=True,
    )

    spec = envelope.to_request_spec()
    round_tripped = ContentRequestEnvelope.from_request_spec(spec)

    assert spec.provider == "openai"
    assert spec.model == "gpt-5"
    assert spec.stream is True
    assert round_tripped.messages[0].blocks[0] == TextBlock("hello")
    assert isinstance(round_tripped.messages[0].blocks[1], ImageBlock)
    assert round_tripped.extra == {"trace": "yes"}


def test_content_response_envelope_round_trips_completion_result() -> None:
    result = CompletionResult(
        content="hello",
        reasoning="reasoning trace",
        usage=Usage(total_tokens=3),
        model="gpt-5-mini",
        finish_reason="stop",
        status=200,
    )

    envelope = ContentResponseEnvelope.from_completion_result(result)
    rebuilt = envelope.to_completion_result()

    assert envelope.message.role is Role.ASSISTANT
    assert envelope.message.blocks[0] == TextBlock("hello")
    assert envelope.reasoning == "reasoning trace"
    assert rebuilt.content == "hello"
    assert rebuilt.reasoning == "reasoning trace"
    assert rebuilt.model == "gpt-5-mini"


def test_content_projection_lossy_mode_degrades_unsupported_blocks_to_text() -> None:
    projection = project_content_blocks(
        [
            ImageBlock("https://example.com/x.png"),
            AudioBlock(audio_url="https://example.com/a.mp3"),
            FileBlock(file_url="https://example.com/doc.pdf"),
            MetadataBlock(data={"ticket": "REL-204", "severity": "medium"}),
        ],
        provider="anthropic",
        mode=ContentHandlingMode.LOSSY,
    )

    assert [block for block in projection.blocks] == [
        TextBlock("[image] https://example.com/x.png"),
        TextBlock("[audio] https://example.com/a.mp3"),
        TextBlock("[file] https://example.com/doc.pdf"),
        TextBlock("Metadata: ticket=REL-204; severity=medium"),
    ]
    assert [degradation.block_type for degradation in projection.degradations] == [
        ContentBlockType.IMAGE,
        ContentBlockType.AUDIO,
        ContentBlockType.FILE,
    ]


def test_content_projection_can_explicitly_skip_metadata_blocks() -> None:
    projection = project_content_blocks(
        [
            TextBlock("hello"),
            MetadataBlock(data={"ticket": "REL-204"}),
        ],
        provider="openai",
        include_metadata=False,
    )

    assert projection.blocks == (TextBlock("hello"),)
    assert projection.degradations == ()


def test_content_projection_strict_mode_raises_for_unsupported_blocks() -> None:
    try:
        project_content_blocks(
            [FileBlock(file_url="https://example.com/doc.pdf")],
            provider="openai",
            mode=ContentHandlingMode.STRICT,
        )
    except UnsupportedContentError as exc:
        assert exc.provider == "openai"
        assert exc.block_type is ContentBlockType.FILE
    else:
        raise AssertionError("expected UnsupportedContentError")
