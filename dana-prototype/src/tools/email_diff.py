"""
email_diff.py

Production-ready utilities to diff two email bodies.

Features:
- Normalization (line endings, trailing whitespace, optional whitespace collapsing)
- Unified diff output (human-friendly, patch-like)
- JSON edit script (machine-friendly opcodes)
- Apply edit script to reconstruct new version from old (useful for verification)
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence


OpTag = Literal["replace", "delete", "insert"]
JsonOp = Dict[str, Any]


@dataclass(frozen=True)
class DiffSettings:
    """
    Settings that control normalization and output behavior.

    - normalize_line_endings:
        Convert CRLF/CR -> LF to avoid false diffs across platforms/clients.
    - strip_trailing_whitespace:
        Removes trailing spaces on each line (very common email/client noise).
    - collapse_whitespace:
        If True, collapses runs of whitespace to a single space *within lines*.
        Use with care: this changes meaning for preformatted content.
    - keep_blank_lines:
        If False, removes consecutive blank lines beyond one (optional hygiene).
    - context_lines:
        Unified diff context lines around changes.
    """
    normalize_line_endings: bool = True
    strip_trailing_whitespace: bool = True
    collapse_whitespace: bool = False
    keep_blank_lines: bool = True
    context_lines: int = 3


def normalize_email_body(text: str, settings: DiffSettings) -> str:
    """
    Normalize an email body for more stable diffs across email clients and platforms.
    """
    if text is None:
        text = ""

    s = text

    if settings.normalize_line_endings:
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    lines = s.split("\n")

    if settings.strip_trailing_whitespace:
        lines = [ln.rstrip() for ln in lines]

    if settings.collapse_whitespace:
        # Collapse whitespace runs inside a line, but preserve leading indentation
        # by only collapsing within the "content" portion after leading spaces/tabs.
        import re

        def _collapse(ln: str) -> str:
            m = re.match(r"^([ \t]*)(.*)$", ln)
            if not m:
                return ln
            indent, rest = m.group(1), m.group(2)
            rest = re.sub(r"\s+", " ", rest).strip()
            return indent + rest

        lines = [_collapse(ln) for ln in lines]

    if not settings.keep_blank_lines:
        compact: List[str] = []
        prev_blank = False
        for ln in lines:
            blank = (ln.strip() == "")
            if blank and prev_blank:
                continue
            compact.append(ln)
            prev_blank = blank
        lines = compact

    return "\n".join(lines)


def unified_diff(
    old_email: str,
    new_email: str,
    *,
    settings: Optional[DiffSettings] = None,
    fromfile: str = "old_email.txt",
    tofile: str = "new_email.txt",
) -> str:
    """
    Generate a unified diff (text).

    Output is suitable for display, logging, or patch-like workflows.
    """
    st = settings or DiffSettings()
    old_n = normalize_email_body(old_email, st).splitlines(keepends=True)
    new_n = normalize_email_body(new_email, st).splitlines(keepends=True)

    diff_iter = difflib.unified_diff(
        old_n,
        new_n,
        fromfile=fromfile,
        tofile=tofile,
        n=st.context_lines,
        lineterm="",
    )
    return "\n".join(diff_iter)


def json_edit_script(
    old_email: str,
    new_email: str,
    *,
    settings: Optional[DiffSettings] = None,
) -> List[JsonOp]:
    """
    Generate a JSON-friendly edit script (SequenceMatcher opcodes) over normalized text.

    Each op is:
      - op: 'replace' | 'delete' | 'insert'
      - a_span: [i1, i2] character offsets in the *normalized* old string
      - b_text: text to insert/replace (empty for delete)
    """
    st = settings or DiffSettings()
    old_n = normalize_email_body(old_email, st)
    new_n = normalize_email_body(new_email, st)

    sm = difflib.SequenceMatcher(a=old_n, b=new_n)
    ops: List[JsonOp] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag not in ("replace", "delete", "insert"):
            # SequenceMatcher only emits these tags, but keep this defensive.
            raise ValueError(f"Unexpected diff tag: {tag}")

        ops.append(
            {
                "op": tag,             # replace | delete | insert
                "a_span": [i1, i2],    # offsets in normalized old
                "b_text": new_n[j1:j2] # replacement/insert payload
            }
        )

    return ops


def apply_json_edit_script(
    old_email: str,
    ops: Sequence[JsonOp],
    *,
    settings: Optional[DiffSettings] = None,
) -> str:
    """
    Apply a JSON edit script to an old email to reconstruct the new version.

    Important:
    - This expects the ops were generated against the *normalized* old email string.
    - If your caller applies against a different 'old', you should detect mismatch
      (e.g., store a hash of normalized old alongside ops).
    """
    st = settings or DiffSettings()
    s = normalize_email_body(old_email, st)

    # Apply from end to start so spans remain valid.
    def _span_start(op: JsonOp) -> int:
        a_span = op.get("a_span")
        if not isinstance(a_span, list) or len(a_span) != 2:
            raise ValueError(f"Invalid a_span: {a_span}")
        return int(a_span[0])

    for op in sorted(ops, key=_span_start, reverse=True):
        tag: str = str(op.get("op"))
        a_span = op.get("a_span")
        b_text = op.get("b_text", "")
        if not isinstance(a_span, list) or len(a_span) != 2:
            raise ValueError(f"Invalid a_span: {a_span}")

        i1, i2 = int(a_span[0]), int(a_span[1])
        if i1 < 0 or i2 < i1 or i2 > len(s):
            raise ValueError(f"Span out of bounds: {a_span} for len={len(s)}")

        if tag == "delete":
            s = s[:i1] + s[i2:]
        elif tag == "insert":
            # For insert, i1==i2 is typical, but not required.
            s = s[:i1] + str(b_text) + s[i1:]
        elif tag == "replace":
            s = s[:i1] + str(b_text) + s[i2:]
        else:
            raise ValueError(f"Unknown op tag: {tag}")

    return s


def generate_email_diffs(
    old_email: str,
    new_email: str,
    *,
    settings: Optional[DiffSettings] = None,
    fromfile: str = "old_email.txt",
    tofile: str = "new_email.txt",
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns both unified diff and JSON edit script.
    """
    st = settings or DiffSettings()
    return {
        "settings": st.__dict__,
        "unified_diff": unified_diff(old_email, new_email, settings=st, fromfile=fromfile, tofile=tofile),
        "json_edit_script": json_edit_script(old_email, new_email, settings=st),
    }


def dumps_json(obj: Any, *, indent: int = 2) -> str:
    """
    Safe JSON serialization helper (UTF-8 friendly).
    """
    return json.dumps(obj, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    # Minimal demo
    v1 = "Hi Sarah,\r\n\r\nThanks for your time.\r\nBest,\r\nAmirMasoud\r\n"
    v2 = "Hi Sarah,\r\n\r\nThanks for your time today.\r\nBest regards,\r\nAmirMasoud\r\n"

    st = DiffSettings(strip_trailing_whitespace=True, context_lines=3)
    out = generate_email_diffs(v1, v2, settings=st, fromfile="v1", tofile="v2")

    print(out["unified_diff"])
    print()
    print(dumps_json(out["json_edit_script"]))

    # Verify apply works
    reconstructed = apply_json_edit_script(v1, out["json_edit_script"], settings=st)
    assert reconstructed == normalize_email_body(v2, st)
