"""
Tests for idempotency module.
"""

import time

from llm_client.idempotency import (
    IdempotencyTracker,
    PendingRequest,
    compute_request_hash,
    generate_idempotency_key,
    get_tracker,
)


class TestIdempotencyKeyGeneration:
    """Test key generation functions."""

    def test_generate_key(self):
        """Test generating unique keys."""
        key1 = generate_idempotency_key()
        key2 = generate_idempotency_key()

        assert key1.startswith("idem_")
        assert key2.startswith("idem_")
        assert key1 != key2

    def test_generate_key_with_prefix(self):
        """Test custom prefix."""
        key = generate_idempotency_key(prefix="req")

        assert key.startswith("req_")

    def test_generate_key_without_timestamp(self):
        """Test key without timestamp."""
        key = generate_idempotency_key(include_timestamp=False)

        # Should be shorter without timestamp
        parts = key.split("_")
        assert len(parts) == 2  # prefix + uuid


class TestRequestHash:
    """Test request hashing."""

    def test_same_input_same_hash(self):
        """Test deterministic hashing."""
        hash1 = compute_request_hash(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
        )
        hash2 = compute_request_hash(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
        )

        assert hash1 == hash2

    def test_different_input_different_hash(self):
        """Test different inputs produce different hashes."""
        hash1 = compute_request_hash(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )
        hash2 = compute_request_hash(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4",
        )

        assert hash1 != hash2

    def test_hash_length(self):
        """Test hash is reasonable length."""
        h = compute_request_hash(
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4",
        )

        assert len(h) == 32  # Truncated SHA256


class TestPendingRequest:
    """Test PendingRequest dataclass."""

    def test_create(self):
        """Test creating pending request."""
        req = PendingRequest(key="test_key")

        assert req.key == "test_key"
        assert req.started_at > 0

    def test_not_expired(self):
        """Test fresh request is not expired."""
        req = PendingRequest(key="test")

        assert not req.is_expired(timeout=60.0)

    def test_expired(self):
        """Test expired request detection."""
        req = PendingRequest(key="test", started_at=time.time() - 120)

        assert req.is_expired(timeout=60.0)


class TestIdempotencyTracker:
    """Test IdempotencyTracker."""

    def test_can_start_new(self):
        """Test can start new request."""
        tracker = IdempotencyTracker()

        assert tracker.can_start("key1")

    def test_cannot_start_duplicate(self):
        """Test cannot start duplicate request."""
        tracker = IdempotencyTracker()

        tracker.start_request("key1")

        assert not tracker.can_start("key1")

    def test_start_request(self):
        """Test starting a request."""
        tracker = IdempotencyTracker()

        assert tracker.start_request("key1")
        assert tracker.is_pending("key1")

        # Can't start again
        assert not tracker.start_request("key1")

    def test_complete_request(self):
        """Test completing a request."""
        tracker = IdempotencyTracker()

        tracker.start_request("key1")
        tracker.complete_request("key1", result={"status": "success"})

        assert not tracker.is_pending("key1")
        assert tracker.has_result("key1")
        assert tracker.get_result("key1") == {"status": "success"}

    def test_fail_request(self):
        """Test failing a request."""
        tracker = IdempotencyTracker()

        tracker.start_request("key1")
        tracker.fail_request("key1")

        assert not tracker.is_pending("key1")
        assert not tracker.has_result("key1")

    def test_expired_cleanup(self):
        """Test expired requests are cleaned up."""
        tracker = IdempotencyTracker(request_timeout=0.1)

        tracker.start_request("key1")
        assert tracker.is_pending("key1")

        # Wait for expiration
        time.sleep(0.15)

        # Should be cleaned up
        assert not tracker.is_pending("key1")
        assert tracker.can_start("key1")

    def test_clear(self):
        """Test clearing all requests."""
        tracker = IdempotencyTracker()

        tracker.start_request("key1")
        tracker.start_request("key2")
        tracker.complete_request("key2", result="done")

        tracker.clear()

        assert tracker.pending_count == 0
        assert tracker.completed_count == 0

    def test_counts(self):
        """Test pending and completed counts."""
        tracker = IdempotencyTracker()

        tracker.start_request("k1")
        tracker.start_request("k2")
        tracker.complete_request("k2", result="done")

        assert tracker.pending_count == 1
        assert tracker.completed_count == 1


class TestGlobalTracker:
    """Test global tracker function."""

    def test_get_tracker(self):
        """Test getting global tracker."""
        tracker = get_tracker()

        assert tracker is not None
        assert isinstance(tracker, IdempotencyTracker)

    def test_same_instance(self):
        """Test returns same instance."""
        t1 = get_tracker()
        t2 = get_tracker()

        assert t1 is t2
