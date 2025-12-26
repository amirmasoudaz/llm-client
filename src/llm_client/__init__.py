"""
Top-level package for the LLM client.

Environment variables are loaded from the nearest `.env` to keep parity with
the original script behavior.
"""
from dotenv import find_dotenv, load_dotenv

# Keep side effect so API keys are loaded on import.
_ = load_dotenv(find_dotenv(), override=True)

from .cache import (
    QdrantCache,
    FSCache,
    HybridRedisPostgreSQLCache,
)
from .client import OpenAIClient
from .exceptions import ResponseTimeoutError
from .models import (
    GPT5,
    GPT5Mini,
    GPT5Nano,
    GPT5Point1,
    GPT5Point2,
    ModelProfile,
    TextEmbedding3Large,
    TextEmbedding3Small,
)
from .rate_limit import Limiter, TokenBucket
from .streaming import PusherStreamer
from .batch_req import RequestManager

__all__ = [
    "OpenAIClient",
    "ModelProfile",
    "GPT5",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Point1",
    "GPT5Point2",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
    "QdrantCache",
    "FSCache",
    "HybridRedisPostgreSQLCache",
    "Limiter",
    "TokenBucket",
    "PusherStreamer",
    "RequestManager",
    "ResponseTimeoutError",
]
