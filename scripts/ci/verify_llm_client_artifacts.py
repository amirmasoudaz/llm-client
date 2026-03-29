from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
import re

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
_ARTIFACT_VERSION_RE = re.compile(r"^llm_client-(?P<version>.+?)(?:-py3-none-any)?(?:\.tar\.gz|\.whl)$")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _artifact_paths() -> tuple[Path, Path]:
    wheels = sorted(DIST.glob("llm_client-*.whl"))
    sdists = sorted(DIST.glob("llm_client-*.tar.gz"))
    _assert(bool(wheels), "no wheel artifact found in dist/")
    _assert(bool(sdists), "no sdist artifact found in dist/")
    return _latest_artifact(wheels), _latest_artifact(sdists)


def _artifact_version(path: Path) -> Version:
    match = _ARTIFACT_VERSION_RE.match(path.name)
    _assert(match is not None, f"unrecognized artifact filename: {path.name}")
    version_text = str(match.group("version"))
    return Version(version_text)


def _latest_artifact(paths: list[Path]) -> Path:
    return max(paths, key=lambda current: _artifact_version(current))


def verify_wheel_contents(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
    expected = {
        "llm_client/__init__.py",
        "llm_client/assets/model_catalog.json",
        "llm_client/assets/model_catalog.schema.json",
        "llm_client/py.typed",
    }
    missing = [name for name in expected if name not in names]
    _assert(not missing, f"wheel missing expected files: {missing}")


def verify_sdist_contents(sdist_path: Path) -> None:
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = set(archive.getnames())
    expected_suffixes = [
        "/pyproject.toml",
        "/llm_client/__init__.py",
        "/llm_client/assets/model_catalog.json",
        "/llm_client/assets/model_catalog.schema.json",
        "/llm_client/py.typed",
    ]
    for suffix in expected_suffixes:
        _assert(
            any(name.endswith(suffix) for name in names),
            f"sdist missing expected file suffix: {suffix}",
        )


def install_and_smoke_test(wheel_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="llm-client-wheel-") as tmpdir:
        smoke = "\n".join(
            [
                "import sys",
                f"wheel_path = {str(wheel_path)!r}",
                f"repo_root = {str(ROOT)!r}",
                "sys.path = [p for p in sys.path if p not in ('', repo_root)]",
                "sys.path.insert(0, wheel_path)",
                "import llm_client",
                "import llm_client.content",
                "import llm_client.providers",
                "import llm_client.budgets",
                "import llm_client.cache",
                "import llm_client.agent",
                "import llm_client.observability",
                "print('llm_client wheel import smoke passed')",
            ]
        )
        subprocess.run(
            [sys.executable, "-c", smoke],
            check=True,
            cwd=tmpdir,
            text=True,
        )


def main() -> int:
    wheel_path, sdist_path = _artifact_paths()
    print(f"[llm_client artifacts] verifying wheel: {wheel_path.name}")
    verify_wheel_contents(wheel_path)
    print(f"[llm_client artifacts] verifying sdist: {sdist_path.name}")
    verify_sdist_contents(sdist_path)
    print("[llm_client artifacts] verifying wheel install smoke")
    install_and_smoke_test(wheel_path)
    print("[llm_client artifacts] verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
