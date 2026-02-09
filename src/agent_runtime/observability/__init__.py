"""
Observability adapters for agent runtime.

This module provides:
- OpenTelemetryAdapter: OTel tracing integration
- MetricsCollector: Built-in metrics collection
"""

from .otel import OpenTelemetryAdapter, OTelConfig

__all__ = [
    "OpenTelemetryAdapter",
    "OTelConfig",
]
