from decimal import Decimal
from functools import cache, lru_cache
from typing import Any, ClassVar

import tiktoken


@cache
def _get_encoder(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class ModelProfile:
    supported_models = {
        "completions": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.2",
        ],
        "embeddings": [
            "text-embedding-3-large",
            "text-embedding-3-small",
        ],
    }

    key: ClassVar[str]
    model_name: ClassVar[str]
    category: ClassVar[str]
    context_window: ClassVar[int]
    rate_limits: ClassVar[dict]
    usage_costs: ClassVar[dict]
    reasoning_model: ClassVar[bool] = False
    reasoning_efforts: ClassVar[list[str]] = []
    default_reasoning_effort: ClassVar[str | None] = None
    function_calling_support: ClassVar[bool] = False
    token_streaming_support: ClassVar[bool] = False
    encoding: ClassVar[str] = "cl100k_base"

    max_output: ClassVar[int | None] = None
    output_dimensions: ClassVar[int | None] = None

    _registry: ClassVar[dict[str, type["ModelProfile"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        key = getattr(cls, "key", None)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{cls.__name__} must define a non-empty string `key`")

        if key in cls._registry:
            raise ValueError(f"Duplicate model key: {key}")

        cls._registry[key] = cls

    @classmethod
    def input_cost(cls, n_tokens: int) -> Decimal:
        return Decimal(n_tokens) * cls.usage_costs["input"]

    @classmethod
    def output_cost(cls, n_tokens: int) -> Decimal:
        if cls.category != "completions":
            return Decimal("0.0")
        return Decimal(n_tokens) * cls.usage_costs["output"]

    @classmethod
    def cached_input_cost(cls, n_tokens: int) -> Decimal:
        return Decimal(n_tokens) * cls.usage_costs["cached_input"]

    @classmethod
    def tokenize(cls, text: str) -> list[int]:
        return _get_encoder(cls.encoding).encode(text)

    @classmethod
    def detokenize(cls, tokens: list[int]) -> str:
        return _get_encoder(cls.encoding).decode(tokens)

    @classmethod
    def count_tokens(cls, context: Any) -> int:
        """
        Count tokens in the given content.

        Uses a hash-based cache for repeated content to avoid
        re-tokenizing the same strings.
        """
        if context is None:
            return 0
        if isinstance(context, str):
            return cls._count_tokens_str(context)
        if isinstance(context, list):
            # list[str]
            if all(isinstance(x, str) for x in context):
                return sum(cls._count_tokens_str(x) for x in context)

            # list[dict] chat-ish
            num_tokens = 0
            for msg in context:
                if isinstance(msg, dict):
                    for v in msg.values():
                        num_tokens += cls._count_tokens_str(str(v))
                else:
                    num_tokens += cls._count_tokens_str(str(msg))
            return num_tokens

        return cls._count_tokens_str(str(context))

    @classmethod
    @lru_cache(maxsize=4096)
    def _count_tokens_str(cls, text: str) -> int:
        """Cached token counting for individual strings."""
        return len(cls.tokenize(text))

    @classmethod
    def parse_usage(cls, usage: dict[str, Any]) -> dict[str, int | Decimal]:
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        parsed_usage: dict[str, int | Decimal] = {
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
        }

        output_tokens: int | None = None
        if "completion_tokens" in usage:
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            parsed_usage["output_tokens"] = output_tokens

        cached_tokens = 0
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = int(prompt_tokens_details.get("cached_tokens", 0) or 0)

        if cached_tokens > 0:
            input_cost_cached = cls.cached_input_cost(cached_tokens)
            input_cost_uncached = cls.input_cost(input_tokens - cached_tokens)
            input_cost = input_cost_cached + input_cost_uncached
            parsed_usage["input_tokens_cached"] = cached_tokens
        else:
            input_cost = cls.input_cost(input_tokens)

        parsed_usage["input_cost"] = input_cost

        output_cost: Decimal | None = None
        if output_tokens is not None:
            output_cost = cls.output_cost(output_tokens)
            parsed_usage["output_cost"] = output_cost

        parsed_usage["total_cost"] = input_cost + (output_cost or Decimal("0"))

        return parsed_usage

    @classmethod
    def get(cls, key: str) -> type["ModelProfile"]:
        try:
            return cls._registry[key]
        except KeyError:
            raise ValueError(f"Unknown model key {key!r}") from None


class GPT5Point2(ModelProfile):
    key = "gpt-5.2"
    model_name = "gpt-5.2-2025-12-11"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("1.75") / Decimal("1000000"),
        "output": Decimal("14.00") / Decimal("1000000"),
        "cached_input": Decimal("0.175") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["none", "minimal", "low", "medium", "high", "xhigh"]
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5Point1(ModelProfile):
    key = "gpt-5.1"
    model_name = "gpt-5.1-2025-11-13"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("1.25") / Decimal("1000000"),
        "output": Decimal("10.00") / Decimal("1000000"),
        "cached_input": Decimal("0.125") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["none", "minimal", "low", "medium", "high"]
    default_reasoning_effort = "none"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5(ModelProfile):
    key = "gpt-5"
    model_name = "gpt-5-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("1.25") / Decimal("1000000"),
        "output": Decimal("10.00") / Decimal("1000000"),
        "cached_input": Decimal("0.125") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5Mini(ModelProfile):
    key = "gpt-5-mini"
    model_name = "gpt-5-mini-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("0.25") / Decimal("1000000"),
        "output": Decimal("2.00") / Decimal("1000000"),
        "cached_input": Decimal("0.025") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5Nano(ModelProfile):
    key = "gpt-5-nano"
    model_name = "gpt-5-nano-2025-08-07"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("0.05") / Decimal("1000000"),
        "output": Decimal("0.40") / Decimal("1000000"),
        "cached_input": Decimal("0.005") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class TextEmbedding3Large(ModelProfile):
    key = "text-embedding-3-large"
    model_name = "text-embedding-3-large"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 3_072
    usage_costs = {"input": Decimal("0.13") / Decimal("1000000")}
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}
    encoding = "cl100k_base"


class TextEmbedding3Small(ModelProfile):
    key = "text-embedding-3-small"
    model_name = "text-embedding-3-small"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {"input": Decimal("0.02") / Decimal("1000000")}
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}
    encoding = "cl100k_base"


class Gemini20Flash(ModelProfile):
    key = "gemini-2.0-flash"
    model_name = "gemini-2.0-flash-exp"
    category = "completions"
    context_window = 1_000_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("0.10") / Decimal("1000000"),
        "output": Decimal("0.40") / Decimal("1000000"),
        "cached_input": Decimal("0.025") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 4_000_000, "req_per_min": 15}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Gemini15Pro(ModelProfile):
    key = "gemini-1.5-pro"
    model_name = "gemini-1.5-pro-002"
    category = "completions"
    context_window = 2_000_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("1.25") / Decimal("1000000"),
        "output": Decimal("5.00") / Decimal("1000000"),
        "cached_input": Decimal("0.3125") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 15}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Claude35Sonnet(ModelProfile):
    key = "claude-3-5-sonnet"
    model_name = "claude-3-5-sonnet-20241022"
    category = "completions"
    context_window = 200_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("15.00") / Decimal("1000000"),
        "cached_input": Decimal("0.30") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 80_000, "req_per_min": 4_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"  # Approximation


class Claude35Haiku(ModelProfile):
    key = "claude-3-5-haiku"
    model_name = "claude-3-5-haiku-20241022"
    category = "completions"
    context_window = 200_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("1.00") / Decimal("1000000"),
        "output": Decimal("5.00") / Decimal("1000000"),
        "cached_input": Decimal("0.10") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 100_000, "req_per_min": 5_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Claude3Opus(ModelProfile):
    key = "claude-3-opus"
    model_name = "claude-3-opus-20240229"
    category = "completions"
    context_window = 200_000
    max_output = 4_096
    usage_costs = {
        "input": Decimal("15.00") / Decimal("1000000"),
        "output": Decimal("75.00") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 40_000, "req_per_min": 2_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Claude45Haiku(ModelProfile):
    """Claude 4.5 Haiku - Fast, cost-effective model."""

    key = "claude-4-5-haiku"
    model_name = "claude-4-5-haiku-20260115"
    category = "completions"
    context_window = 200_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("1.00") / Decimal("1000000"),
        "output": Decimal("5.00") / Decimal("1000000"),
        "cached_input": Decimal("0.10") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 120_000, "req_per_min": 6_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Claude45Sonnet(ModelProfile):
    """Claude 4.5 Sonnet - Balanced performance and intelligence."""

    key = "claude-4-5-sonnet"
    model_name = "claude-4-5-sonnet-20260115"
    category = "completions"
    context_window = 200_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("15.00") / Decimal("1000000"),
        "cached_input": Decimal("0.30") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 100_000, "req_per_min": 5_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Claude45Opus(ModelProfile):
    """Claude 4.5 Opus - Most capable model with extended thinking."""

    key = "claude-4-5-opus"
    model_name = "claude-4-5-opus-20260115"
    category = "completions"
    context_window = 200_000
    max_output = 8_192
    usage_costs = {
        "input": Decimal("5.00") / Decimal("1000000"),
        "output": Decimal("25.00") / Decimal("1000000"),
        "cached_input": Decimal("0.50") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 60_000, "req_per_min": 3_000}
    function_calling_support = True
    token_streaming_support = True
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    encoding = "cl100k_base"


class Gemini3Flash(ModelProfile):
    """Gemini 3 Flash - Fast and efficient model."""

    key = "gemini-3-flash"
    model_name = "gemini-3-flash-20260120"
    category = "completions"
    context_window = 1_000_000
    max_output = 65_536
    usage_costs = {
        "input": Decimal("0.50") / Decimal("1000000"),
        "output": Decimal("3.00") / Decimal("1000000"),
        "cached_input": Decimal("0.125") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 5_000_000, "req_per_min": 20}
    function_calling_support = True
    token_streaming_support = True
    encoding = "cl100k_base"


class Gemini3Pro(ModelProfile):
    """Gemini 3 Pro - High-capability model with 1M context."""

    key = "gemini-3-pro"
    model_name = "gemini-3-pro-20260120"
    category = "completions"
    context_window = 1_000_000
    max_output = 65_536
    usage_costs = {
        "input": Decimal("2.00") / Decimal("1000000"),
        "output": Decimal("12.00") / Decimal("1000000"),
        "cached_input": Decimal("0.50") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 3_000_000, "req_per_min": 15}
    function_calling_support = True
    token_streaming_support = True
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    encoding = "cl100k_base"


__all__ = [
    "ModelProfile",
    # OpenAI
    "GPT5",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Point1",
    "GPT5Point2",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
    # Gemini
    "Gemini20Flash",
    "Gemini15Pro",
    "Gemini3Flash",
    "Gemini3Pro",
    # Claude
    "Claude35Sonnet",
    "Claude35Haiku",
    "Claude3Opus",
    "Claude45Haiku",
    "Claude45Sonnet",
    "Claude45Opus",
]
