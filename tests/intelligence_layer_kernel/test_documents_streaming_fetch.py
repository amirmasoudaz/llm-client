from __future__ import annotations

from pathlib import Path

from blake3 import blake3

from intelligence_layer_kernel.operators.implementations import documents_common


def test_fetch_attachment_bytes_streams_to_disk_and_hashes(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "cv.txt"
    content = b"CV\nExperience\nSkills\n"
    source.write_bytes(content)

    monkeypatch.setenv("IL_DOCUMENT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("IL_ATTACHMENT_MAX_BYTES", "1048576")

    fetched = documents_common.fetch_attachment_bytes(
        {
            "file_path": str(source),
            "mime": "text/plain",
            "object_uri": "s3://bucket/cv.txt",
        }
    )

    assert fetched["status"] == "downloaded"
    assert fetched["bytes"] is None
    assert isinstance(fetched.get("stream_path"), str)
    assert fetched["size_bytes"] == len(content)
    assert fetched["hash_hex"] == blake3(content).hexdigest()

    stream_path = Path(str(fetched["stream_path"]))
    assert stream_path.exists()
    assert stream_path.read_bytes() == content

    documents_common.remove_cached_file(str(stream_path))
    assert not stream_path.exists()


def test_fetch_attachment_bytes_blocks_disallowed_mime(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "script.js"
    source.write_text("console.log('x')", encoding="utf-8")

    monkeypatch.setenv("IL_DOCUMENT_CACHE_DIR", str(tmp_path / "cache"))

    fetched = documents_common.fetch_attachment_bytes(
        {
            "file_path": str(source),
            "mime": "application/javascript",
            "object_uri": "s3://bucket/script.js",
        }
    )

    assert fetched["status"] == "blocked_mime"
    assert fetched["bytes"] is None
    assert fetched["stream_path"] is None


def test_fetch_attachment_bytes_enforces_size_limit(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_bytes(b"a" * 4096)

    monkeypatch.setenv("IL_DOCUMENT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("IL_ATTACHMENT_MAX_BYTES", "1024")

    fetched = documents_common.fetch_attachment_bytes(
        {
            "file_path": str(source),
            "mime": "text/plain",
            "object_uri": "s3://bucket/large.txt",
        }
    )

    assert fetched["status"] == "too_large"
    assert fetched["bytes"] is None
    assert fetched["stream_path"] is None
    assert int(fetched["size_bytes"]) > 1024


def test_extract_text_from_bytes_handles_pdf_docx_and_text(monkeypatch) -> None:
    monkeypatch.setattr(documents_common, "_extract_pdf_text", lambda _data: ("PDF extracted text", 2))
    monkeypatch.setattr(documents_common, "_extract_docx_text", lambda _data: "DOCX extracted text")

    pdf_text, pdf_pages, pdf_strategy = documents_common.extract_text_from_bytes(
        b"%PDF-1.4\n...",
        mime="application/pdf",
        file_name="resume.pdf",
    )
    assert pdf_strategy == "pdf"
    assert pdf_text == "PDF extracted text"
    assert pdf_pages == 2

    docx_text, docx_pages, docx_strategy = documents_common.extract_text_from_bytes(
        b"PK\x03\x04docx-bytes",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_name="statement.docx",
    )
    assert docx_strategy == "docx"
    assert docx_text == "DOCX extracted text"
    assert docx_pages is None

    text_text, text_pages, text_strategy = documents_common.extract_text_from_bytes(
        "Plain text content".encode("utf-8"),
        mime="text/plain",
        file_name="notes.txt",
    )
    assert text_strategy == "text"
    assert text_text == "Plain text content"
    assert text_pages is None
