import pytest

from llm_client.content import AudioBlock, ContentHandlingMode, FileBlock, ImageBlock, MetadataBlock, TextBlock, UnsupportedContentError
from llm_client.providers.openai import OpenAIProvider
from llm_client.providers.types import Message


def test_openai_provider_translates_canonical_content_blocks_to_chat_parts() -> None:
    message = Message.user(
        [
            TextBlock("hello"),
            ImageBlock("https://example.com/img.png", detail="high"),
            AudioBlock(data="YmFzZTY0", mime_type="audio/wav"),
            FileBlock(file_url="https://example.com/doc.pdf", name="doc.pdf"),
            MetadataBlock(data={"ignored": True}),
        ]
    )

    payload = OpenAIProvider._messages_to_api_format([message])

    assert payload == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}},
                {"type": "input_audio", "input_audio": {"data": "YmFzZTY0", "format": "wav"}},
                {"type": "text", "text": "[file] https://example.com/doc.pdf"},
                {"type": "text", "text": "Metadata: ignored=True"},
            ],
        }
    ]


def test_openai_provider_strict_content_mode_rejects_unsupported_file_blocks() -> None:
    with pytest.raises(UnsupportedContentError):
        OpenAIProvider._messages_to_api_format(
            [Message.user([FileBlock(file_url="https://example.com/doc.pdf")])],
            content_mode=ContentHandlingMode.STRICT,
        )
