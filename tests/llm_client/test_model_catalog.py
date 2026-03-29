import json

import pytest

from llm_client.config.provider import AnthropicConfig, GoogleConfig, OpenAIConfig
from llm_client.model_catalog import (
    DEFAULT_MODEL_CATALOG_PATH,
    MODEL_CATALOG_OVERRIDE_PATH_ENV,
    clear_model_catalog_cache,
    get_default_model_catalog,
    infer_provider_for_model,
    load_model_catalog,
    metadata_from_profile,
)
from llm_client.models import GPT5, ModelProfile, TextEmbedding3Small
from llm_client.provider_registry import get_default_provider_registry


def test_model_catalog_loads_asset_backed_metadata() -> None:
    catalog = get_default_model_catalog()

    gpt5 = catalog.get("gpt-5")

    assert str(DEFAULT_MODEL_CATALOG_PATH) == catalog.source
    assert gpt5.provider == "openai"
    assert gpt5.reasoning is True
    assert gpt5.tool_calling is True
    assert gpt5.streaming is True
    assert gpt5.structured_outputs is True
    assert gpt5.vision_input is True
    assert gpt5.context_window >= 400_000


def test_model_catalog_filters_by_provider_category_and_capability() -> None:
    catalog = get_default_model_catalog()

    google_models = catalog.list(provider="google", category="completions", structured_outputs=True)
    embedding_models = catalog.list(provider="openai", category="embeddings")

    assert any(item.key == "gemini-2.0-flash" for item in google_models)
    assert [item.key for item in embedding_models] == ["text-embedding-3-large", "text-embedding-3-small"]


def test_model_catalog_resolves_provider_defaults() -> None:
    catalog = get_default_model_catalog()

    assert catalog.default_for_provider("openai").key == "gpt-5"
    assert catalog.default_for_provider("openai", category="embeddings").key == "text-embedding-3-small"
    assert catalog.default_for_provider("google").key == "gemini-2.0-flash"
    assert catalog.default_for_provider("anthropic").key == "claude-sonnet-4"


def test_model_catalog_override_support_changes_defaults(tmp_path, monkeypatch) -> None:
    override_path = tmp_path / "model_catalog.override.json"
    override_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "openai": {"completions": "gpt-5-mini"},
                },
                "models": [
                    {
                        "key": "gpt-5",
                        "deprecated": True,
                        "replacement": "gpt-5-mini",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(MODEL_CATALOG_OVERRIDE_PATH_ENV, str(override_path))
    clear_model_catalog_cache()
    get_default_provider_registry.cache_clear()
    try:
        catalog = get_default_model_catalog()

        assert catalog.default_for_provider("openai").key == "gpt-5-mini"
        assert catalog.get("gpt-5").deprecated is True
        assert catalog.get("gpt-5").replacement == "gpt-5-mini"
        assert OpenAIConfig().default_model == "gpt-5-mini"
        assert get_default_provider_registry().get("openai").default_model == "gpt-5-mini"
    finally:
        monkeypatch.delenv(MODEL_CATALOG_OVERRIDE_PATH_ENV, raising=False)
        clear_model_catalog_cache()
        get_default_provider_registry.cache_clear()


def test_model_catalog_schema_validation_rejects_bad_documents(tmp_path) -> None:
    bad_path = tmp_path / "invalid_model_catalog.json"
    bad_path.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {},
                "models": [
                    {
                        "key": "broken-model",
                        "model_name": "broken-model",
                        "category": "completions",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    clear_model_catalog_cache()
    with pytest.raises(ValueError, match="Invalid model catalog document"):
        load_model_catalog(catalog_path=bad_path)


def test_model_catalog_asset_matches_profile_snapshot() -> None:
    catalog = get_default_model_catalog()
    expected_keys = sorted(ModelProfile._registry)

    assert [item.key for item in catalog.list()] == expected_keys

    for key in expected_keys:
        asset = catalog.get(key)
        derived = metadata_from_profile(ModelProfile.get(key))

        assert asset.to_dict() == derived.to_dict()


def test_model_metadata_helpers_infer_provider_and_serialize() -> None:
    embedding = metadata_from_profile(TextEmbedding3Small)

    assert infer_provider_for_model("gpt-5-mini") == "openai"
    assert infer_provider_for_model("gemini-3-pro") == "google"
    assert infer_provider_for_model("claude-4-5-sonnet") == "anthropic"
    assert embedding.to_dict()["provider"] == "openai"
    assert metadata_from_profile(GPT5).key == "gpt-5"


def test_provider_configs_use_catalog_defaults() -> None:
    clear_model_catalog_cache()
    get_default_provider_registry.cache_clear()
    assert OpenAIConfig().default_model == "gpt-5"
    assert AnthropicConfig().default_model == "claude-sonnet-4"
    assert GoogleConfig().default_model == "gemini-2.0-flash"
