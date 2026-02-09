"""Tests for embedding cache parity in ExecutionEngine."""

from __future__ import annotations

import pytest

from llm_client import ExecutionEngine
from llm_client.providers.types import EmbeddingResult, Usage


class TestEmbeddingCacheKey:
    """Test _embed_cache_key generation."""

    def test_consistent_key(self, mock_provider, memory_cache):
        """Same inputs should produce same cache key."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        key1 = engine._embed_cache_key(["hello", "world"])
        key2 = engine._embed_cache_key(["hello", "world"])
        
        assert key1 == key2

    def test_different_inputs_different_key(self, mock_provider, memory_cache):
        """Different inputs should produce different keys."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        key1 = engine._embed_cache_key(["hello"])
        key2 = engine._embed_cache_key(["world"])
        
        assert key1 != key2


class TestEmbeddingCacheSerialization:
    """Test embedding result serialization."""

    def test_embedding_to_cache(self):
        """Should serialize embedding result to cache dict."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            usage=Usage(input_tokens=10, output_tokens=0, total_tokens=10),
            model="text-embedding-3-small",
        )
        
        cached = ExecutionEngine._embedding_to_cache(result)
        
        assert cached["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert cached["usage"]["input_tokens"] == 10
        assert cached["model"] == "text-embedding-3-small"

    def test_cached_to_embedding_result(self):
        """Should deserialize cache dict to embedding result."""
        cached = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
            "model": "test-model",
        }
        
        result = ExecutionEngine._cached_to_embedding_result(cached)
        
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.usage.input_tokens == 5
        assert result.model == "test-model"

    def test_roundtrip(self):
        """Serialization should be reversible."""
        original = EmbeddingResult(
            embeddings=[[1.0, 2.0]],
            usage=Usage(input_tokens=100, output_tokens=0, total_tokens=100),
            model="test",
        )
        
        cached = ExecutionEngine._embedding_to_cache(original)
        restored = ExecutionEngine._cached_to_embedding_result(cached)
        
        assert restored.embeddings == original.embeddings
        assert restored.usage.input_tokens == original.usage.input_tokens
        assert restored.model == original.model


class TestEmbedWithCache:
    """Test ExecutionEngine.embed() with caching."""

    @pytest.mark.asyncio
    async def test_embed_without_cache(self, mock_provider, memory_cache):
        """Should work without caching enabled."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        result = await engine.embed(["hello"])
        
        assert result is not None
        assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_embed_with_cache_miss(self, mock_provider, memory_cache):
        """Should call provider on cache miss."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        result = await engine.embed(["hello"], cache_response=True)
        
        assert result is not None
        assert provider._embed_count == 1

    @pytest.mark.asyncio
    async def test_embed_with_cache_hit(self, mock_provider, memory_cache):
        """Should return cached result on hit."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        # First call - cache miss
        result1 = await engine.embed(["hello"], cache_response=True)
        assert provider._embed_count == 1
        
        # Second call - cache hit
        result2 = await engine.embed(["hello"], cache_response=True)
        assert provider._embed_count == 1  # No additional call
        
        # Results should be equivalent
        assert result1.embeddings == result2.embeddings

    @pytest.mark.asyncio
    async def test_embed_cache_different_inputs(self, mock_provider, memory_cache):
        """Different inputs should not share cache."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        await engine.embed(["hello"], cache_response=True)
        await engine.embed(["world"], cache_response=True)
        
        assert provider._embed_count == 2  # Both should call provider

    @pytest.mark.asyncio
    async def test_embed_no_cache_backend(self, mock_provider):
        """Should work when no cache is configured."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider)  # No cache
        
        result = await engine.embed(["hello"], cache_response=True)
        
        assert result is not None  # Should not raise


class TestEmbedRouting:
    """Test embedding provider routing."""

    @pytest.mark.asyncio
    async def test_embed_uses_default_provider(self, mock_provider, memory_cache):
        """Should use default provider."""
        provider = mock_provider()
        engine = ExecutionEngine(provider=provider, cache=memory_cache)
        
        await engine.embed(["test"])
        
        # Verify provider was called
        assert provider._embed_count >= 1
