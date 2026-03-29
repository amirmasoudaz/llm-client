from __future__ import annotations

from pathlib import Path


def test_public_api_map_document_exists_and_mentions_core_namespaces() -> None:
    doc = Path("docs/llm-client-public-api-v1.md")

    assert doc.exists()
    text = doc.read_text()
    assert "## Stable Namespaces" in text
    assert "llm_client.providers" in text
    assert "llm_client.types" in text
    assert "llm_client.content" in text
    assert "llm_client.observability" in text
    assert "llm_client.compat" in text
    assert "llm_client.advanced" in text
    assert "llm_client.memory" in text


def test_readme_links_to_public_api_map() -> None:
    readme = Path("llm_client/README.md").read_text()

    assert "docs/llm-client-public-api-v1.md" in readme


def test_architecture_and_adoption_docs_exist_and_are_linked() -> None:
    readme = Path("llm_client/README.md").read_text()
    architecture = Path("docs/llm-client-architecture.md")
    matrix = Path("docs/llm-client-extraction-matrix.md")
    agent_runtime_notes = Path("docs/llm-client-adoption-notes-agent-runtime.md")
    intelligence_layer_notes = Path("docs/llm-client-adoption-notes-intelligence-layer.md")
    threat_model = Path("docs/llm-client-threat-model.md")
    secure_defaults = Path("docs/llm-client-secure-deployment-defaults.md")

    assert architecture.exists()
    assert matrix.exists()
    assert agent_runtime_notes.exists()
    assert intelligence_layer_notes.exists()
    assert threat_model.exists()
    assert secure_defaults.exists()
    assert "docs/llm-client-architecture.md" in readme
    assert "docs/llm-client-extraction-matrix.md" in readme
    assert "docs/llm-client-adoption-notes-agent-runtime.md" in readme
    assert "docs/llm-client-adoption-notes-intelligence-layer.md" in readme
    assert "docs/llm-client-threat-model.md" in readme
    assert "docs/llm-client-secure-deployment-defaults.md" in readme
