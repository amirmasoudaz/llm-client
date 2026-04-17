from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
GUIDE_INDEX = ROOT / "docs" / "llm-client-guides-index.md"


EXPECTED_GUIDES = [
    "llm-client-guides-index.md",
    "llm-client-build-and-recipes-guide.md",
    "llm-client-provider-setup-guide.md",
    "llm-client-routing-and-failover-guide.md",
    "llm-client-tool-runtime-guide.md",
    "llm-client-structured-outputs-guide.md",
    "llm-client-context-and-memory-guide.md",
    "llm-client-observability-and-redaction-guide.md",
    "llm-client-migration-from-direct-sdk-guide.md",
]


def test_guides_exist() -> None:
    missing = [name for name in EXPECTED_GUIDES if not (ROOT / "docs" / name).exists()]
    assert missing == []


def test_readme_links_guides_index() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "docs/llm-client-guides-index.md" in readme


def test_guide_index_lists_expected_guides() -> None:
    guide_index = GUIDE_INDEX.read_text(encoding="utf-8")
    for name in EXPECTED_GUIDES[1:]:
        assert name in guide_index


def test_guide_index_excludes_archived_transition_docs() -> None:
    guide_index = GUIDE_INDEX.read_text(encoding="utf-8")
    assert "llm-client-modernization-roadmap-2026-03-09.md" not in guide_index
    assert "llm-client-repo-split-guidance.md" not in guide_index
    assert "llm-client-release-notes-1.0.0-rc1.md" not in guide_index
