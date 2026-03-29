"""
Canonical model metadata catalog.

This module provides an asset-backed metadata API so model capabilities,
pricing, defaults, and lifecycle metadata can be shared across projects
without hard-coding everything into Python classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional runtime dependency
    Draft202012Validator = None

from .models import ModelProfile


MODEL_CATALOG_PATH_ENV = "LLM_CLIENT_MODEL_CATALOG_PATH"
MODEL_CATALOG_OVERRIDE_PATH_ENV = "LLM_CLIENT_MODEL_CATALOG_OVERRIDE_PATH"
ASSETS_DIR = Path(__file__).with_name("assets")
DEFAULT_MODEL_CATALOG_PATH = ASSETS_DIR / "model_catalog.json"
DEFAULT_MODEL_CATALOG_SCHEMA_PATH = ASSETS_DIR / "model_catalog.schema.json"


@dataclass(frozen=True)
class ModelMetadata:
    key: str
    provider: str
    model_name: str
    category: str
    context_window: int
    max_output: int | None
    output_dimensions: int | None
    encoding: str
    reasoning: bool
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str | None
    tool_calling: bool
    streaming: bool
    structured_outputs: bool
    vision_input: bool
    audio_input: bool
    file_input: bool
    deprecated: bool
    replacement: str | None
    rate_limits: dict[str, int]
    usage_costs: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model_name": self.model_name,
            "category": self.category,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "output_dimensions": self.output_dimensions,
            "encoding": self.encoding,
            "reasoning": self.reasoning,
            "reasoning_efforts": list(self.reasoning_efforts),
            "default_reasoning_effort": self.default_reasoning_effort,
            "tool_calling": self.tool_calling,
            "streaming": self.streaming,
            "structured_outputs": self.structured_outputs,
            "vision_input": self.vision_input,
            "audio_input": self.audio_input,
            "file_input": self.file_input,
            "deprecated": self.deprecated,
            "replacement": self.replacement,
            "rate_limits": dict(self.rate_limits),
            "usage_costs": dict(self.usage_costs),
        }


class ModelCatalog:
    def __init__(
        self,
        items: list[ModelMetadata],
        *,
        defaults: dict[str, dict[str, str]] | None = None,
        source: str | None = None,
    ) -> None:
        self._items = {item.key: item for item in items}
        self._defaults = {
            str(provider).strip().lower(): {
                str(category).strip().lower(): str(model).strip()
                for category, model in categories.items()
                if str(model).strip()
            }
            for provider, categories in (defaults or {}).items()
        }
        self.source = source

    def get(self, key: str) -> ModelMetadata:
        try:
            return self._items[key]
        except KeyError:
            raise ValueError(f"Unknown model key {key!r}") from None

    def list(
        self,
        *,
        provider: str | None = None,
        category: str | None = None,
        reasoning: bool | None = None,
        tool_calling: bool | None = None,
        streaming: bool | None = None,
        structured_outputs: bool | None = None,
    ) -> list[ModelMetadata]:
        provider_norm = str(provider or "").strip().lower() or None
        category_norm = str(category or "").strip().lower() or None
        results: list[ModelMetadata] = []
        for item in sorted(self._items.values(), key=lambda current: (current.provider, current.key)):
            if provider_norm is not None and item.provider != provider_norm:
                continue
            if category_norm is not None and item.category != category_norm:
                continue
            if reasoning is not None and item.reasoning != reasoning:
                continue
            if tool_calling is not None and item.tool_calling != tool_calling:
                continue
            if streaming is not None and item.streaming != streaming:
                continue
            if structured_outputs is not None and item.structured_outputs != structured_outputs:
                continue
            results.append(item)
        return results

    def default_key_for_provider(self, provider: str, *, category: str = "completions") -> str | None:
        provider_name = str(provider or "").strip().lower()
        category_name = str(category or "").strip().lower() or "completions"
        if not provider_name:
            return None
        configured = self._defaults.get(provider_name, {}).get(category_name)
        if configured:
            return configured
        matches = self.list(provider=provider_name, category=category_name)
        return matches[0].key if matches else None

    def default_for_provider(self, provider: str, *, category: str = "completions") -> ModelMetadata | None:
        key = self.default_key_for_provider(provider, category=category)
        if not key:
            return None
        try:
            return self.get(key)
        except ValueError:
            return None

    def to_document(self) -> dict[str, object]:
        return {
            "version": 1,
            "defaults": {
                provider: dict(categories)
                for provider, categories in sorted(self._defaults.items())
            },
            "models": [item.to_dict() for item in self.list()],
        }


def infer_provider_for_model(model_key: str) -> str:
    key = str(model_key or "").strip().lower()
    if key.startswith(("gpt-", "text-embedding-")):
        return "openai"
    if key.startswith("gemini-"):
        return "google"
    if key.startswith("claude-"):
        return "anthropic"
    return "unknown"


def _default_capability_flags(profile: type[ModelProfile]) -> dict[str, bool]:
    provider = infer_provider_for_model(profile.key)
    is_completion = profile.category == "completions"
    if provider == "openai":
        return {
            "structured_outputs": is_completion,
            "vision_input": is_completion,
            "audio_input": is_completion,
            "file_input": is_completion,
        }
    if provider == "google":
        return {
            "structured_outputs": is_completion,
            "vision_input": is_completion,
            "audio_input": is_completion,
            "file_input": is_completion,
        }
    if provider == "anthropic":
        return {
            "structured_outputs": False,
            "vision_input": is_completion,
            "audio_input": False,
            "file_input": is_completion,
        }
    return {
        "structured_outputs": False,
        "vision_input": False,
        "audio_input": False,
        "file_input": False,
    }


def metadata_from_profile(profile: type[ModelProfile]) -> ModelMetadata:
    usage_costs = {
        str(name): float(value if isinstance(value, Decimal) else Decimal(str(value)))
        for name, value in dict(getattr(profile, "usage_costs", {})).items()
    }
    rate_limits = {str(name): int(value) for name, value in dict(getattr(profile, "rate_limits", {})).items()}
    defaults = _default_capability_flags(profile)
    return ModelMetadata(
        key=profile.key,
        provider=infer_provider_for_model(profile.key),
        model_name=profile.model_name,
        category=profile.category,
        context_window=profile.context_window,
        max_output=getattr(profile, "max_output", None),
        output_dimensions=getattr(profile, "output_dimensions", None),
        encoding=getattr(profile, "encoding", "cl100k_base"),
        reasoning=bool(getattr(profile, "reasoning_model", False)),
        reasoning_efforts=tuple(getattr(profile, "reasoning_efforts", []) or []),
        default_reasoning_effort=getattr(profile, "default_reasoning_effort", None),
        tool_calling=bool(getattr(profile, "function_calling_support", False)),
        streaming=bool(getattr(profile, "token_streaming_support", False)),
        structured_outputs=defaults["structured_outputs"],
        vision_input=defaults["vision_input"],
        audio_input=defaults["audio_input"],
        file_input=defaults["file_input"],
        deprecated=False,
        replacement=None,
        rate_limits=rate_limits,
        usage_costs=usage_costs,
    )


def _fallback_catalog_document_from_profiles() -> dict[str, object]:
    items = [metadata_from_profile(profile).to_dict() for _, profile in sorted(ModelProfile._registry.items())]
    return {
        "version": 1,
        "defaults": {
            "openai": {
                "completions": "gpt-5",
                "embeddings": "text-embedding-3-small",
            },
            "google": {
                "completions": "gemini-2.0-flash",
            },
            "anthropic": {
                "completions": "claude-sonnet-4",
            },
        },
        "models": items,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_catalog_document(document: dict[str, Any], schema_path: Path) -> None:
    if Draft202012Validator is None:
        return
    schema = _load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.path) or "<root>"
    raise ValueError(f"Invalid model catalog document at {location}: {error.message}")


def _normalize_defaults(defaults: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for provider, categories in (defaults or {}).items():
        provider_name = str(provider).strip().lower()
        if not provider_name or not isinstance(categories, dict):
            continue
        normalized[provider_name] = {}
        for category, model_key in categories.items():
            category_name = str(category).strip().lower()
            model_name = str(model_key or "").strip()
            if category_name and model_name:
                normalized[provider_name][category_name] = model_name
    return normalized


def _merge_catalog_documents(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    merged_models: dict[str, dict[str, Any]] = {
        str(item["key"]): dict(item)
        for item in base.get("models", [])
        if isinstance(item, dict) and str(item.get("key") or "").strip()
    }
    for item in override.get("models", []) if isinstance(override.get("models"), list) else []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        merged = dict(merged_models.get(key, {}))
        merged.update(item)
        merged_models[key] = merged
    merged_defaults = _normalize_defaults(base.get("defaults"))
    for provider, categories in _normalize_defaults(override.get("defaults")).items():
        merged_defaults.setdefault(provider, {}).update(categories)
    return {
        "version": int(override.get("version") or base.get("version") or 1),
        "defaults": merged_defaults,
        "models": [merged_models[key] for key in sorted(merged_models)],
    }


def _catalog_from_document(document: dict[str, Any], *, source: str | None = None) -> ModelCatalog:
    items: list[ModelMetadata] = []
    for raw in document.get("models", []):
        item = ModelMetadata(
            key=str(raw["key"]),
            provider=str(raw["provider"]).strip().lower(),
            model_name=str(raw["model_name"]),
            category=str(raw["category"]).strip().lower(),
            context_window=int(raw["context_window"]),
            max_output=None if raw.get("max_output") is None else int(raw["max_output"]),
            output_dimensions=None if raw.get("output_dimensions") is None else int(raw["output_dimensions"]),
            encoding=str(raw.get("encoding") or "cl100k_base"),
            reasoning=bool(raw.get("reasoning", False)),
            reasoning_efforts=tuple(str(item) for item in (raw.get("reasoning_efforts") or [])),
            default_reasoning_effort=(
                None if raw.get("default_reasoning_effort") is None else str(raw["default_reasoning_effort"])
            ),
            tool_calling=bool(raw.get("tool_calling", False)),
            streaming=bool(raw.get("streaming", False)),
            structured_outputs=bool(raw.get("structured_outputs", False)),
            vision_input=bool(raw.get("vision_input", False)),
            audio_input=bool(raw.get("audio_input", False)),
            file_input=bool(raw.get("file_input", False)),
            deprecated=bool(raw.get("deprecated", False)),
            replacement=None if raw.get("replacement") is None else str(raw["replacement"]),
            rate_limits={str(name): int(value) for name, value in dict(raw.get("rate_limits") or {}).items()},
            usage_costs={str(name): float(value) for name, value in dict(raw.get("usage_costs") or {}).items()},
        )
        items.append(item)
    return ModelCatalog(items, defaults=_normalize_defaults(document.get("defaults")), source=source)


def _resolved_catalog_paths(
    catalog_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    base = catalog_path or os.getenv(MODEL_CATALOG_PATH_ENV)
    override = override_path or os.getenv(MODEL_CATALOG_OVERRIDE_PATH_ENV)
    return (
        Path(base).expanduser().resolve() if base else DEFAULT_MODEL_CATALOG_PATH,
        Path(override).expanduser().resolve() if override else None,
    )


@lru_cache(maxsize=8)
def load_model_catalog(
    catalog_path: str | Path | None = None,
    *,
    override_path: str | Path | None = None,
) -> ModelCatalog:
    base_path, resolved_override_path = _resolved_catalog_paths(catalog_path, override_path)
    if base_path is not None and base_path.exists():
        document = _load_json(base_path)
    else:
        document = _fallback_catalog_document_from_profiles()
    if resolved_override_path is not None and resolved_override_path.exists():
        document = _merge_catalog_documents(document, _load_json(resolved_override_path))
    _validate_catalog_document(document, DEFAULT_MODEL_CATALOG_SCHEMA_PATH)
    source = str(base_path) if base_path is not None else "<embedded>"
    return _catalog_from_document(document, source=source)


def get_default_model_catalog() -> ModelCatalog:
    return load_model_catalog()


def clear_model_catalog_cache() -> None:
    load_model_catalog.cache_clear()


def get_provider_default_model(provider: str, *, category: str = "completions") -> str | None:
    catalog = get_default_model_catalog()
    return catalog.default_key_for_provider(provider, category=category)


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_MODEL_CATALOG_PATH",
    "DEFAULT_MODEL_CATALOG_SCHEMA_PATH",
    "MODEL_CATALOG_OVERRIDE_PATH_ENV",
    "MODEL_CATALOG_PATH_ENV",
    "ModelMetadata",
    "ModelCatalog",
    "clear_model_catalog_cache",
    "get_default_model_catalog",
    "get_provider_default_model",
    "infer_provider_for_model",
    "load_model_catalog",
    "metadata_from_profile",
]
