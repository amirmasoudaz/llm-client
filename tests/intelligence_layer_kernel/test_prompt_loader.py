from __future__ import annotations

from pathlib import Path

import pytest
from blake3 import blake3

from intelligence_layer_kernel.prompts.loader import PromptTemplateLoader

pytest.importorskip("jinja2")


def _write_template(base: Path, rel_path: str, content: str) -> None:
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prompt_loader_renders_versioned_operator_template(tmp_path: Path) -> None:
    template_source = "Subject: {{ email.subject }}\nBody: {{ email.body }}"
    _write_template(
        tmp_path,
        "Email.ReviewDraft/1.0.0/review_draft.j2",
        template_source,
    )
    loader = PromptTemplateLoader(base_path=tmp_path)

    rendered = loader.render(
        "Email.ReviewDraft/1.0.0/review_draft",
        {"email": {"subject": "Hello", "body": "World"}},
    )

    assert rendered.template_id == "Email.ReviewDraft/1.0.0/review_draft.j2"
    assert rendered.text == "Subject: Hello\nBody: World"
    assert rendered.template_hash == blake3(template_source.encode("utf-8")).hexdigest()


def test_prompt_loader_uses_strict_undefined_variables(tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        "Email.ReviewDraft/1.0.0/review_draft.j2",
        "Subject: {{ email.subject }} {{ missing_field }}",
    )
    loader = PromptTemplateLoader(base_path=tmp_path)

    with pytest.raises(Exception) as exc_info:
        loader.render(
            "Email.ReviewDraft/1.0.0/review_draft",
            {"email": {"subject": "Hello"}},
        )

    assert "missing_field" in str(exc_info.value)


def test_prompt_loader_rejects_non_versioned_template_ids(tmp_path: Path) -> None:
    _write_template(tmp_path, "Email.ReviewDraft/1.0.0/review_draft.j2", "ok")
    loader = PromptTemplateLoader(base_path=tmp_path)

    with pytest.raises(ValueError):
        loader.render("email_review_draft.v1", {"email": {"subject": "x", "body": "y"}})
