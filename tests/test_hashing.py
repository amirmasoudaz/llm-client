"""Tests for the hashing module."""

import pytest

from llm_client.hashing import cache_key, compute_hash, content_hash, int_hash


class TestContentHash:
    """Tests for content_hash function."""

    def test_deterministic_same_dict(self) -> None:
        """Same dict produces same hash."""
        obj = {"a": 1, "b": 2}
        assert content_hash(obj) == content_hash(obj)

    def test_deterministic_different_key_order(self) -> None:
        """Dict order doesn't affect hash."""
        a = {"b": 1, "a": {"z": 3, "y": 2}}
        b = {"a": {"y": 2, "z": 3}, "b": 1}
        assert content_hash(a) == content_hash(b)

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hash."""
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_returns_64_char_hex(self) -> None:
        """Hash is 64 character hex string."""
        result = content_hash({"test": "data"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestCacheKey:
    """Tests for cache_key function."""

    def test_consistent_output(self) -> None:
        """Same inputs produce same cache key."""
        key1 = cache_key("chat.completions", {"model": "gpt-5"})
        key2 = cache_key("chat.completions", {"model": "gpt-5"})
        assert key1 == key2

    def test_different_api_different_key(self) -> None:
        """Different API endpoints produce different keys."""
        key1 = cache_key("chat.completions", {"model": "gpt-5"})
        key2 = cache_key("responses", {"model": "gpt-5"})
        assert key1 != key2

    def test_different_params_different_key(self) -> None:
        """Different params produce different keys."""
        key1 = cache_key("chat.completions", {"model": "gpt-5"})
        key2 = cache_key("chat.completions", {"model": "gpt-5-nano"})
        assert key1 != key2


class TestIntHash:
    """Tests for int_hash function."""

    def test_returns_int(self) -> None:
        """Returns an integer."""
        result = int_hash("test")
        assert isinstance(result, int)

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        assert int_hash("hello") == int_hash("hello")

    def test_different_input_different_hash(self) -> None:
        """Different input produces different output."""
        assert int_hash("hello") != int_hash("world")

    def test_non_negative(self) -> None:
        """Result is non-negative (unsigned)."""
        result = int_hash("test")
        assert result >= 0


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_blake3_default(self) -> None:
        """Blake3 is the default algorithm."""
        h1 = compute_hash("test")
        h2 = compute_hash("test", algorithm="blake3")
        assert h1 == h2

    def test_sha256(self) -> None:
        """SHA256 produces valid hash."""
        result = compute_hash("test", algorithm="sha256")
        assert len(result) == 64

    def test_md5(self) -> None:
        """MD5 produces valid hash."""
        result = compute_hash("test", algorithm="md5")
        assert len(result) == 32

    def test_truncate(self) -> None:
        """Truncation works correctly."""
        result = compute_hash("test", truncate=16)
        assert len(result) == 16

    def test_bytes_input(self) -> None:
        """Accepts bytes input."""
        h1 = compute_hash("test")
        h2 = compute_hash(b"test")
        assert h1 == h2

    def test_invalid_algorithm(self) -> None:
        """Invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            compute_hash("test", algorithm="invalid")  # type: ignore
