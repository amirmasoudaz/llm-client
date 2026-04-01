from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.content import (
    ContentHandlingMode,
    FileBlock,
    Message,
    TextBlock,
    content_blocks_to_openai_chat_content,
    content_blocks_to_openai_responses_content,
    prepare_file_block,
    project_content_blocks,
)
from llm_client.request_builders import build_request_spec
from llm_client.providers.openai import OpenAIProvider


def test_prepare_file_block_reads_local_file_and_computes_transport_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "release-notes.txt"
    file_path.write_text("release candidate ready", encoding="utf-8")

    block = prepare_file_block(FileBlock(file_path=str(file_path)))

    assert block.file_path == str(file_path)
    assert block.name == "release-notes.txt"
    assert block.mime_type == "text/plain"
    assert isinstance(block.data, str)
    assert block.sha256 is not None
    assert block.size_bytes == len("release candidate ready".encode("utf-8"))


def test_request_builder_prepares_local_file_blocks_for_cacheable_engine_requests(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello from a local file", encoding="utf-8")

    spec = build_request_spec(
        provider=type("OpenAIProviderFake", (), {"model_name": "gpt-5-mini"})(),
        messages=[Message.user([TextBlock("Read this file"), FileBlock(file_path=str(file_path))])],
    )

    payload = spec.to_dict()
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    file_payload = content[1]
    assert file_payload["file_path"] == str(file_path)
    assert file_payload["sha256"]
    assert file_payload["size_bytes"] == len("hello from a local file".encode("utf-8"))


def test_openai_chat_content_still_degrades_file_blocks_without_native_file_transport() -> None:
    payload = content_blocks_to_openai_chat_content(
        [FileBlock(file_url="https://example.com/release.pdf", extracted_text="release content")],
        mode=ContentHandlingMode.LOSSY,
    )

    assert payload == [{"type": "text", "text": "release content"}]


def test_openai_responses_content_uses_native_file_transport_for_inline_or_referenced_files(tmp_path: Path) -> None:
    file_path = tmp_path / "brief.txt"
    file_path.write_text("mission control status", encoding="utf-8")

    payload = content_blocks_to_openai_responses_content(
        [
            TextBlock("Summarize the attached file."),
            FileBlock(file_path=str(file_path)),
            FileBlock(file_id="file-123", name="uploaded.pdf"),
        ],
        mode=ContentHandlingMode.STRICT,
    )

    assert payload[0] == {"type": "input_text", "text": "Summarize the attached file."}
    assert payload[1]["type"] == "input_file"
    assert payload[1]["filename"] == "brief.txt"
    assert "file_data" in payload[1]
    assert payload[2] == {"type": "input_file", "file_id": "file-123"}


def test_openai_responses_content_uses_native_file_transport_for_url_only_files() -> None:
    payload = content_blocks_to_openai_responses_content(
        [FileBlock(file_url="https://example.com/guide.pdf", extracted_text="guide body text")],
        mode=ContentHandlingMode.LOSSY,
    )

    assert payload == [{"type": "input_file", "file_url": "https://example.com/guide.pdf"}]


def test_non_native_providers_use_extracted_text_sidecar_for_file_fallback() -> None:
    projection = project_content_blocks(
        [FileBlock(file_url="https://example.com/guide.pdf", extracted_text="guide body text")],
        provider="anthropic",
        mode=ContentHandlingMode.LOSSY,
    )

    assert projection.blocks == (TextBlock("guide body text"),)
    assert projection.degradations[0].reason == "provider does not support file content"


def test_openai_responses_content_strict_mode_accepts_url_only_files() -> None:
    payload = content_blocks_to_openai_responses_content(
        [FileBlock(file_url="https://example.com/guide.pdf")],
        mode=ContentHandlingMode.STRICT,
    )

    assert payload == [{"type": "input_file", "file_url": "https://example.com/guide.pdf"}]


def test_openai_provider_messages_to_api_format_uses_responses_file_transport(tmp_path: Path) -> None:
    file_path = tmp_path / "incident.txt"
    file_path.write_text("incident notes", encoding="utf-8")

    payload = OpenAIProvider._messages_to_api_format(
        [Message.user([TextBlock("Read the file"), FileBlock(file_path=str(file_path))])],
        responses_api=True,
    )

    content = payload[0]["content"]
    assert content[0] == {"type": "input_text", "text": "Read the file"}
    assert content[1]["type"] == "input_file"
    assert content[1]["filename"] == "incident.txt"
