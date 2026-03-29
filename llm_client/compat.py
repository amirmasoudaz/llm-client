"""
Compatibility namespace for legacy llm-client APIs.

This module collects APIs retained for backward compatibility so new projects
can prefer the stable package namespaces while existing code can migrate
incrementally.
"""

from .client import OpenAIClient
from .exceptions import ResponseTimeoutError

__all__ = [
    "OpenAIClient",
    "ResponseTimeoutError",
]
