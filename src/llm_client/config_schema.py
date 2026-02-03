"""
JSON schemas for configuration validation.
"""

PROVIDER_SCHEMA = {
    "type": "object",
    "properties": {
        "api_key": {"type": ["string", "null"]},
        "base_url": {"type": ["string", "null"]},
        "organization": {"type": ["string", "null"]},
        "timeout": {"type": "number", "minimum": 0.1},
        "max_retries": {"type": "integer", "minimum": 0},
        "retry_backoff": {"type": "number", "minimum": 0.0},
        "default_model": {"type": ["string", "null"]},
        "default_temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "default_max_tokens": {"type": ["integer", "null"], "minimum": 1},
    },
    "additionalProperties": True,  # Allow provider-specific fields
}

OPENAI_SCHEMA = {
    "allOf": [
        PROVIDER_SCHEMA,
        {
            "properties": {
                "use_responses_api": {"type": "boolean"},
            }
        },
    ]
}

ANTHROPIC_SCHEMA = {
    "allOf": [
        PROVIDER_SCHEMA,
        {
            "properties": {
                "max_thinking_tokens": {"type": ["integer", "null"], "minimum": 1},
            }
        },
    ]
}

CACHE_SCHEMA = {
    "type": "object",
    "properties": {
        "backend": {"type": "string", "enum": ["none", "fs", "pg_redis", "qdrant"]},
        "enabled": {"type": "boolean"},
        "default_collection": {"type": ["string", "null"]},
        "ttl_seconds": {"type": ["integer", "null"], "minimum": 1},
        "cache_errors": {"type": "boolean"},
        "only_cache_ok": {"type": "boolean"},
        # FSCache
        "cache_dir": {"type": "string"},
        # RedisPG
        "pg_dsn": {"type": "string"},
        "redis_url": {"type": "string"},
        "redis_ttl_seconds": {"type": "integer", "minimum": 1},
        "compress": {"type": "boolean"},
        "compression_level": {"type": "integer", "minimum": 0, "maximum": 9},
        # Qdrant
        "qdrant_url": {"type": "string"},
        "qdrant_api_key": {"type": ["string", "null"]},
    },
    "required": ["backend"],
    "allOf": [
        {
            "if": {"properties": {"backend": {"const": "fs"}}},
            "then": {"required": ["cache_dir"]},
        },
        {
            "if": {"properties": {"backend": {"const": "pg_redis"}}},
            "then": {"required": ["pg_dsn", "redis_url"]},
        },
        {
            "if": {"properties": {"backend": {"const": "qdrant"}}},
            "then": {"required": ["qdrant_url"]},
        },
    ],
}

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "max_turns": {"type": "integer", "minimum": 1},
        "max_tool_calls_per_turn": {"type": "integer", "minimum": 1},
        "parallel_tool_execution": {"type": "boolean"},
        "tool_timeout": {"type": "number", "minimum": 0.1},
        "max_tool_output_chars": {"type": ["integer", "null"], "minimum": 1},
        "max_tokens": {"type": ["integer", "null"], "minimum": 1},
        "reserve_tokens": {"type": "integer", "minimum": 0},
        "stop_on_tool_error": {"type": "boolean"},
        "include_tool_errors_in_context": {"type": "boolean"},
        "stream_tool_calls": {"type": "boolean"},
        "batch_concurrency": {"type": "integer", "minimum": 1},
    },
}

LOGGING_SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
        "format": {"type": "string", "enum": ["text", "json"]},
        "log_file": {"type": ["string", "null"]},
        "include_timestamp": {"type": "boolean"},
        "include_trace_id": {"type": "boolean"},
        "log_requests": {"type": "boolean"},
        "log_responses": {"type": "boolean"},
        "log_tool_calls": {"type": "boolean"},
        "log_usage": {"type": "boolean"},
        "redact_api_keys": {"type": "boolean"},
    },
}

METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "provider": {"type": "string", "enum": ["none", "prometheus", "otel"]},
        "prometheus_port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "otel_endpoint": {"type": ["string", "null"]},
        "otel_service_name": {"type": "string"},
    },
}

RATE_LIMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "requests_per_minute": {"type": "integer", "minimum": 1},
        "tokens_per_minute": {"type": "integer", "minimum": 1},
        "wait_on_limit": {"type": "boolean"},
        "max_wait_seconds": {"type": "number", "minimum": 0.0},
    },
}

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "openai": OPENAI_SCHEMA,
        "anthropic": ANTHROPIC_SCHEMA,
        "cache": CACHE_SCHEMA,
        "agent": AGENT_SCHEMA,
        "logging": LOGGING_SCHEMA,
        "metrics": METRICS_SCHEMA,
        "rate_limit": RATE_LIMIT_SCHEMA,
    },
}
