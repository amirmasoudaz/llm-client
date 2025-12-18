from typing import ClassVar, Dict, Union, Any, Type
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=None)
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
    function_calling_support: ClassVar[bool] = False
    token_streaming_support: ClassVar[bool] = False
    encoding: ClassVar[str] = "cl100k_base"

    max_output: ClassVar[int | None] = None
    output_dimensions: ClassVar[int | None] = None

    _registry: ClassVar[Dict[str, Type["ModelProfile"]]] = {}

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)

        key = getattr(cls, "key", None)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{cls.__name__} must define a non-empty string `key`")

        if key in cls._registry:
            raise ValueError(f"Duplicate model key: {key}")

        cls._registry[key] = cls

    @classmethod
    def input_cost(cls, n_tokens: int) -> float:
        return n_tokens * cls.usage_costs["input"]

    @classmethod
    def output_cost(cls, n_tokens: int) -> float:
        if cls.category != "completions":
            return 0.0
        return n_tokens * cls.usage_costs["output"]

    @classmethod
    def cached_input_cost(cls, n_tokens: int) -> float:
        return n_tokens * cls.usage_costs["cached_input"]

    @classmethod
    def tokenize(cls, text: str) -> list[int]:
        return _get_encoder(cls.encoding).encode(text)

    @classmethod
    def detokenize(cls, tokens: list[int]) -> str:
        return _get_encoder(cls.encoding).decode(tokens)

    @classmethod
    def count_tokens(cls, context: Any) -> int:
        if context is None:
            return 0
        if isinstance(context, str):
            return len(cls.tokenize(context))
        if isinstance(context, list):
            # list[str]
            if all(isinstance(x, str) for x in context):
                return sum(len(cls.tokenize(x)) for x in context)

            # list[dict] chat-ish
            num_tokens = 0
            for msg in context:
                if isinstance(msg, dict):
                    for v in msg.values():
                        num_tokens += len(cls.tokenize(str(v)))
                else:
                    num_tokens += len(cls.tokenize(str(msg)))
            return num_tokens

        return len(cls.tokenize(str(context)))

    @classmethod
    def parse_usage(cls, usage: dict) -> dict:
        parsed_usage: Dict[str, Union[int, float]] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if "completion_tokens" in usage:
            parsed_usage["output_tokens"] = usage["completion_tokens"]
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"].get("cached_tokens", 0) > 0:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            input_cost_cached = cls.cached_input_cost(cached_tokens)
            input_cost_uncached = cls.input_cost(parsed_usage["input_tokens"] - cached_tokens)
            input_cost = input_cost_cached + input_cost_uncached
            parsed_usage["input_tokens_cached"] = cached_tokens
        else:
            input_cost = cls.input_cost(parsed_usage["input_tokens"])

        parsed_usage["input_cost"] = input_cost
        if "output_tokens" in parsed_usage:
            parsed_usage["output_cost"] = cls.output_cost(parsed_usage["output_tokens"])
        parsed_usage["total_cost"] = parsed_usage["input_cost"] + parsed_usage.get("output_cost", 0)

        return parsed_usage

    @classmethod
    def get(cls, key: str) -> Type["ModelProfile"]:
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
        "input": 1.75 / 1_000_000,
        "output": 14.00 / 1_000_000,
        "cached_input": 0.175 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
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
        "input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
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
        "input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
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
        "input": 0.25 / 1_000_000,
        "output": 2.00 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
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
        "input": 0.05 / 1_000_000,
        "output": 0.40 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class TextEmbedding3Large(ModelProfile):
    key = "text-embedding-3-large"
    model_name = "text-embedding-3-large"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 3_072
    usage_costs = {"input": 0.13 / 1_000_000}
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}
    encoding = "cl100k_base"


class TextEmbedding3Small(ModelProfile):
    key = "text-embedding-3-small"
    model_name = "text-embedding-3-small"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {"input": 0.02 / 1_000_000}
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 5_000}
    encoding = "cl100k_base"


__all__ = [
    "ModelProfile",
    "GPT5",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Point1",
    "GPT5Point2",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
]
