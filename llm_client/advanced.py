"""
Advanced and compatibility-oriented helpers for llm-client.

This namespace groups lower-level utilities and integration surfaces that are
useful in some projects but are not part of the preferred standalone-package
API for most new applications. Prefer the stable namespaces such as
`llm_client.providers`, `llm_client.engine`, `llm_client.agent`,
`llm_client.tools`, `llm_client.types`, and `llm_client.observability` unless
you specifically need these lower-level capabilities.
"""

from .container import (
    Container,
    ServiceRegistry,
    create_agent,
    create_anthropic_provider,
    create_cache,
    create_google_provider,
    create_openai_provider,
    create_provider,
    get_container,
    set_container,
)
from .hashing import cache_key, compute_hash, content_hash, int_hash
from .idempotency import (
    IdempotencyTracker,
    PendingRequest,
    compute_request_hash,
    generate_idempotency_key,
    get_tracker,
)
from .perf import (
    FingerprintCache,
    clear_fingerprint_cache,
    fingerprint,
    fingerprint_messages,
    get_fingerprint,
)
from .serialization import (
    cached_stable_json_dumps,
    canonicalize,
    fast_json_dumps,
    fast_json_loads,
    obj_to_hashable,
    stable_json_dumps,
)
from .streaming import (
    BufferingAdapter,
    CallbackAdapter,
    PusherStreamer,
    SSEAdapter,
    StreamAdapter,
    collect_stream,
    format_sse_event,
    stream_to_string,
)

__all__ = [
    "ServiceRegistry",
    "create_openai_provider",
    "create_anthropic_provider",
    "create_google_provider",
    "create_provider",
    "create_cache",
    "create_agent",
    "Container",
    "get_container",
    "set_container",
    "generate_idempotency_key",
    "compute_request_hash",
    "PendingRequest",
    "IdempotencyTracker",
    "get_tracker",
    "compute_hash",
    "content_hash",
    "cache_key",
    "int_hash",
    "fingerprint",
    "fingerprint_messages",
    "FingerprintCache",
    "get_fingerprint",
    "clear_fingerprint_cache",
    "canonicalize",
    "stable_json_dumps",
    "fast_json_dumps",
    "fast_json_loads",
    "cached_stable_json_dumps",
    "obj_to_hashable",
    "StreamAdapter",
    "SSEAdapter",
    "CallbackAdapter",
    "BufferingAdapter",
    "PusherStreamer",
    "format_sse_event",
    "collect_stream",
    "stream_to_string",
]
