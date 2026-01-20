"""
Tests for cache backends.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


class TestFSCache:
    """Tests for filesystem cache backend."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    async def test_write_and_read(self, temp_cache_dir):
        """Test basic write and read operations."""
        from llm_client.cache import FSCache, FSCacheConfig
        
        config = FSCacheConfig(
            dir=temp_cache_dir,
            client_type="test",
        )
        cache = FSCache(config)
        await cache.ensure_ready()
        
        # Write - must have error='OK' to be considered success
        await cache.write(
            effective_key="test_key",
            response={"content": "test response", "error": "OK"},
            model_name="gpt-5-nano",
        )
        
        # Read
        result = await cache.read("test_key")
        
        assert result is not None
        assert result["content"] == "test response"
        
        await cache.close()
    
    async def test_exists(self, temp_cache_dir):
        """Test exists check."""
        from llm_client.cache import FSCache, FSCacheConfig
        
        config = FSCacheConfig(dir=temp_cache_dir, client_type="test")
        cache = FSCache(config)
        await cache.ensure_ready()
        
        assert not await cache.exists("nonexistent")
        
        await cache.write(
            effective_key="exists_test",
            response={"data": "value", "error": "OK"},
            model_name="test-model",
        )
        
        assert await cache.exists("exists_test")
        
        await cache.close()
    
    async def test_collections(self, temp_cache_dir):
        """Test collection isolation."""
        from llm_client.cache import FSCache, FSCacheConfig
        
        config = FSCacheConfig(
            dir=temp_cache_dir,
            client_type="test",
            default_collection="default",
        )
        cache = FSCache(config)
        await cache.ensure_ready()
        
        # Write to different collections - must have error='OK'
        await cache.write(
            effective_key="shared_key",
            response={"collection": "A", "error": "OK"},
            model_name="test",
            collection="collection_a",
        )
        await cache.write(
            effective_key="shared_key",
            response={"collection": "B", "error": "OK"},
            model_name="test",
            collection="collection_b",
        )
        
        # Read from each collection
        result_a = await cache.read("shared_key", collection="collection_a")
        result_b = await cache.read("shared_key", collection="collection_b")
        
        assert result_a["collection"] == "A"
        assert result_b["collection"] == "B"
        
        await cache.close()
    
    async def test_resolve_key_normal(self, temp_cache_dir):
        """Test key resolution in normal mode."""
        from llm_client.cache import FSCache, FSCacheConfig
        
        config = FSCacheConfig(dir=temp_cache_dir, client_type="test")
        cache = FSCache(config)
        await cache.ensure_ready()
        
        key, can_read = await cache.resolve_key(
            identifier="my_key",
            rewrite_cache=False,
            regen_cache=False,
        )
        
        assert key == "my_key"
        assert can_read is True
        
        await cache.close()
    
    async def test_resolve_key_regen(self, temp_cache_dir):
        """Test key resolution with regen_cache=True."""
        from llm_client.cache import FSCache, FSCacheConfig
        
        config = FSCacheConfig(dir=temp_cache_dir, client_type="test")
        cache = FSCache(config)
        await cache.ensure_ready()
        
        key, can_read = await cache.resolve_key(
            identifier="my_key",
            rewrite_cache=False,
            regen_cache=True,  # Don't read from cache
        )
        
        assert key == "my_key"
        assert can_read is False  # Should not read existing
        
        await cache.close()


class TestCacheCore:
    """Tests for CacheCore wrapper."""
    
    async def test_get_cached_miss(self, memory_cache):
        """Test cache miss returns None for response."""
        from llm_client.cache import CacheCore
        
        core = CacheCore(backend=memory_cache)
        await core.ensure_ready()
        
        result = await core.get_cached(
            identifier="nonexistent",
            rewrite_cache=False,
            regen_cache=False,
        )
        
        # CacheCore returns (response, key) tuple
        if isinstance(result, tuple):
            response, key = result
            assert response is None
        else:
            assert result is None
    
    async def test_put_and_get_cached(self, memory_cache):
        """Test put followed by get."""
        from llm_client.cache import CacheCore
        
        core = CacheCore(backend=memory_cache)
        await core.ensure_ready()
        
        # Response must have error='OK' to be returned by only_ok=True (default)
        await core.put_cached(
            identifier="test_id",
            rewrite_cache=False,
            regen_cache=False,
            response={"status": 200, "content": "cached", "error": "OK"},
            model_name="test-model",
            log_errors=True,
        )
        
        result = await core.get_cached(
            identifier="test_id",
            rewrite_cache=False,
            regen_cache=False,
        )
        
        # CacheCore returns (response, key) tuple
        if isinstance(result, tuple):
            response, key = result
            assert response is not None
            assert response["content"] == "cached"
        else:
            assert result is not None
            assert result["content"] == "cached"
    
    async def test_get_cached_only_ok(self, memory_cache):
        """Test that only_ok=True filters non-200 responses."""
        from llm_client.cache import CacheCore
        
        core = CacheCore(backend=memory_cache)
        await core.ensure_ready()
        
        # Cache an error response
        await core.put_cached(
            identifier="error_response",
            rewrite_cache=False,
            regen_cache=False,
            response={"status": 500, "error": "Server error"},
            model_name="test",
            log_errors=True,
        )
        
        # Should not return with only_ok=True
        result = await core.get_cached(
            identifier="error_response",
            rewrite_cache=False,
            regen_cache=False,
            only_ok=True,
        )
        
        # CacheCore returns (response, key) tuple or None
        if isinstance(result, tuple):
            response, key = result
            assert response is None
        else:
            assert result is None
    
    async def test_no_backend_returns_none(self):
        """Test that no backend gracefully returns None."""
        from llm_client.cache import CacheCore
        
        core = CacheCore(backend=None)
        
        result = await core.get_cached(
            identifier="anything",
            rewrite_cache=False,
            regen_cache=False,
        )
        
        # CacheCore returns (response, key) tuple or None
        if isinstance(result, tuple):
            response, key = result
            assert response is None
        else:
            assert result is None


class TestInMemoryCache:
    """Tests for the in-memory cache fixture."""
    
    async def test_basic_operations(self, memory_cache):
        """Test basic in-memory cache operations."""
        await memory_cache.write(
            effective_key="key1",
            response={"data": "value1"},
            model_name="test",
        )
        
        result = await memory_cache.read("key1")
        assert result["data"] == "value1"
        
        exists = await memory_cache.exists("key1")
        assert exists
        
        not_exists = await memory_cache.exists("key2")
        assert not not_exists
    
    async def test_resolve_key_rewrite(self, memory_cache):
        """Test rewrite creates new suffix."""
        # First, write something with the base key
        await memory_cache.write(
            effective_key="base_key",
            response={"version": 1},
            model_name="test",
        )
        
        # Resolve with rewrite should give new key
        key, can_read = await memory_cache.resolve_key(
            identifier="base_key",
            rewrite_cache=True,
            regen_cache=False,
        )
        
        assert key != "base_key"  # Should have suffix
        assert key.startswith("base_key_")
        assert can_read is False
