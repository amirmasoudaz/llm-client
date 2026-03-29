from __future__ import annotations

import random
import string

from llm_client.providers.openai import OpenAIProvider
from llm_client.structured import _openai_safe_schema_name, _sanitize_json_schema_for_openai


def _rand_word(rng: random.Random, *, min_len: int = 1, max_len: int = 8) -> str:
    length = rng.randint(min_len, max_len)
    alphabet = string.ascii_letters + string.digits + " .-_/:$"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _rand_schema(rng: random.Random, depth: int = 0):
    if depth > 2:
        return rng.choice([_rand_word(rng), rng.randint(0, 10), {"type": "string"}])
    choice = rng.randint(0, 4)
    if choice == 0:
        return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"urn:{_rand_word(rng)}", "type": "string"}
    if choice == 1:
        return {rng.choice(["properties", "$defs", "items", "oneOf"]): _rand_schema(rng, depth + 1)}
    if choice == 2:
        return [_rand_schema(rng, depth + 1) for _ in range(rng.randint(1, 3))]
    if choice == 3:
        return {
            "type": "object",
            "properties": {
                _rand_word(rng): _rand_schema(rng, depth + 1),
                _rand_word(rng): _rand_schema(rng, depth + 1),
            },
            "required": [_rand_word(rng)],
        }
    return {"enum": [_rand_word(rng), _rand_word(rng)], "$id": f"urn:{_rand_word(rng)}"}


def _contains_key(value, forbidden: str) -> bool:
    if isinstance(value, dict):
        if forbidden in value:
            return True
        return any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_fuzz_openai_safe_schema_name_always_matches_contract() -> None:
    rng = random.Random(1337)

    for _ in range(200):
        raw = _rand_word(rng, min_len=0, max_len=120)
        safe = _openai_safe_schema_name(raw)
        assert safe
        assert len(safe) <= 64
        assert all(ch.isalnum() or ch in {"_", "-"} for ch in safe)


def test_fuzz_openai_json_schema_sanitizer_removes_ids_and_schema_keys() -> None:
    rng = random.Random(20260318)

    for _ in range(100):
        schema = _rand_schema(rng)
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {"value": schema}}
        sanitized = _sanitize_json_schema_for_openai(schema)
        assert isinstance(sanitized, dict)
        assert not _contains_key(sanitized, "$schema")
        assert not _contains_key(sanitized, "$id")


def test_fuzz_openai_function_schema_sanitizer_returns_valid_object_shape() -> None:
    rng = random.Random(4242)

    for _ in range(100):
        schema = _rand_schema(rng)
        if not isinstance(schema, dict):
            schema = {"oneOf": [schema], "type": "object"}
        sanitized = OpenAIProvider._sanitize_openai_function_parameters_schema(schema)
        assert isinstance(sanitized, dict)
        assert sanitized.get("type") == "object"
        assert isinstance(sanitized.get("properties"), dict)
        assert sanitized["properties"]
        assert "oneOf" not in sanitized
        assert "$schema" not in sanitized
        assert "$id" not in sanitized
