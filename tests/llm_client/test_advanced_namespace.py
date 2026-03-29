from __future__ import annotations

from llm_client import advanced


def test_advanced_namespace_exposes_low_level_compat_and_helper_surfaces() -> None:
    assert "Container" in advanced.__all__
    assert "IdempotencyTracker" in advanced.__all__
    assert "compute_hash" in advanced.__all__
    assert "fingerprint" in advanced.__all__
    assert "stable_json_dumps" in advanced.__all__
    assert "BufferingAdapter" in advanced.__all__


def test_advanced_namespace_resolves_expected_symbols() -> None:
    assert advanced.compute_hash is not None
    assert advanced.generate_idempotency_key(prefix="x", include_timestamp=False).startswith("x_")
    assert advanced.cache_key("chat.completions", {"q": "hello"})
    assert advanced.stable_json_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'
