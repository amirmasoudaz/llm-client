from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
GUIDE_INDEX = ROOT / "docs" / "llm-client-guides-index.md"
PY_TYPED = ROOT / "llm_client" / "py.typed"

EXPECTED_DOCS = [
    "llm-client-installation-matrix.md",
    "llm-client-packaging-readiness.md",
    "llm-client-changelog-process.md",
    "llm-client-semver-policy.md",
    "llm-client-support-policy.md",
    "llm-client-release-automation.md",
]

EXPECTED_SCRIPTS = [
    "scripts/ci/run_llm_client_examples.py",
    "scripts/ci/verify_llm_client_artifacts.py",
]

EXPECTED_OSS_FILES = [
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
]


def test_pyproject_declares_standalone_package_metadata() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "llm-client"
    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert "authors" in project
    assert "maintainers" in project
    assert "classifiers" in project
    assert "keywords" in project
    extras = project["optional-dependencies"]
    for key in [
        "anthropic",
        "google",
        "postgres",
        "mysql",
        "redis",
        "qdrant",
        "adapters",
        "telemetry",
        "performance",
        "server",
        "dev",
        "all",
    ]:
        assert key in extras


def test_docs_exist() -> None:
    missing = [name for name in EXPECTED_DOCS if not (ROOT / "docs" / name).exists()]
    assert missing == []


def test_release_docs_and_scripts_exist() -> None:
    missing_scripts = [name for name in EXPECTED_SCRIPTS if not (ROOT / name).exists()]
    assert missing_scripts == []
    assert not (ROOT / ".github" / "workflows" / "llm-client-package-ci.yml").exists()
    assert not (ROOT / ".github" / "workflows" / "llm-client-publish.yml").exists()


def test_typing_marker_and_governance_files_exist() -> None:
    assert PY_TYPED.exists()
    missing = [name for name in EXPECTED_OSS_FILES if not (ROOT / name).exists()]
    assert missing == []


def test_readme_and_guide_index_reference_packaging_docs() -> None:
    readme = README.read_text(encoding="utf-8")
    guide_index = GUIDE_INDEX.read_text(encoding="utf-8")
    assert "docs/llm-client-installation-matrix.md" in readme
    assert "docs/llm-client-release-automation.md" in readme
    assert "llm-client-installation-matrix.md" in guide_index
    assert "llm-client-semver-policy.md" in guide_index


def test_release_automation_doc_references_local_validation_steps() -> None:
    release_automation = (ROOT / "docs" / "llm-client-release-automation.md").read_text(
        encoding="utf-8"
    )
    assert "run_llm_client_examples.py" not in release_automation
    assert "GitHub-hosted CI" in release_automation
    assert "twine check" in release_automation
    assert "Build wheel and sdist locally" in release_automation


def test_packaging_inventory_does_not_depend_on_archived_transition_docs() -> None:
    guide_index = GUIDE_INDEX.read_text(encoding="utf-8")
    assert "llm-client-modernization-roadmap-2026-03-09.md" not in guide_index
    assert "llm-client-final-stage-release-checklist-2026-03-24.md" not in guide_index
