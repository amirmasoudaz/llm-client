from __future__ import annotations

import argparse
import re
from pathlib import Path


_PACKAGING_LOCAL_REF = re.compile(r"^\s*packaging\s*@\s*file://", re.IGNORECASE)
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")


def _extract_requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(("-", "--")):
        return None
    match = _REQUIREMENT_NAME.match(stripped)
    if match is None:
        return None
    return match.group(1).lower()


def _normalize_line(
    line: str,
    *,
    packaging_version: str,
    drop_packages: set[str],
) -> str | None:
    stripped = line.strip()
    if _PACKAGING_LOCAL_REF.match(stripped):
        return f"packaging=={packaging_version}"
    requirement_name = _extract_requirement_name(line)
    if requirement_name and requirement_name in drop_packages:
        return None
    return line.rstrip("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize requirements files for CI/container builds by replacing "
            "local path references with portable pinned versions."
        )
    )
    parser.add_argument(
        "--input",
        default="requirements.txt",
        help="Path to source requirements file.",
    )
    parser.add_argument(
        "--output",
        default="requirements.normalized.txt",
        help="Path to write normalized requirements file.",
    )
    parser.add_argument(
        "--packaging-version",
        default="25.0",
        help="Version to use when replacing local packaging references.",
    )
    parser.add_argument(
        "--drop-package",
        action="append",
        default=[],
        help="Package name to omit from the normalized output. Can be provided multiple times.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    source_path = Path(args.input)
    target_path = Path(args.output)
    drop_packages = {str(name).strip().lower() for name in args.drop_package if str(name).strip()}

    lines = source_path.read_text(encoding="utf-8").splitlines()
    normalized: list[str] = []
    for line in lines:
        updated = _normalize_line(
            line,
            packaging_version=args.packaging_version,
            drop_packages=drop_packages,
        )
        if updated is None:
            continue
        normalized.append(updated)

    target_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    print(f"normalized_requirements={target_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
