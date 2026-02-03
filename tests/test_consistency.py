"""
Tests for codebase consistency and registration.
"""

from llm_client.config import GoogleConfig, get_settings
from llm_client.models import Claude35Sonnet, ModelProfile


def test_anthropic_models_registered():
    """Verify Anthropic models are registered in ModelProfile."""
    # This should not raise
    model = ModelProfile.get("claude-3-5-sonnet")
    assert model is Claude35Sonnet
    assert model.key == "claude-3-5-sonnet"

    assert ModelProfile.get("claude-3-5-haiku")
    assert ModelProfile.get("claude-3-opus")


def test_google_config_in_settings():
    """Verify GoogleConfig is available in Settings."""
    settings = get_settings()
    assert hasattr(settings, "google")
    assert isinstance(settings.google, GoogleConfig)
    assert settings.google.default_model == "gemini-2.0-flash"


def test_google_config_env_loading():
    """Verify GoogleConfig loads from environment."""
    import os
    from unittest.mock import patch

    from llm_client.config.settings import Settings

    with patch.dict(os.environ, {"LLM_GOOGLE_API_KEY": "test-key", "LLM_GOOGLE_MODEL": "test-model"}):
        settings = Settings.from_env()
        assert settings.google.api_key == "test-key"
        assert settings.google.default_model == "test-model"
