from decimal import Decimal
from functools import cache, lru_cache
import re
from typing import Any, ClassVar

import logging
import tiktoken


logger = logging.getLogger("llm_client.models")


class _FallbackEncoding:
    """Best-effort tokenizer fallback when tiktoken encodings cannot be loaded.

    This is intentionally approximate (used for counting/budget estimates).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, text: str) -> list[int]:
        data = str(text or "")
        if not data:
            return []
        # Rough heuristic: ~4 characters per token for English-ish text.
        approx = max(1, (len(data) + 3) // 4)
        return [0] * approx

    def decode(self, tokens: list[int]) -> str:
        _ = tokens
        return ""


@cache
def _get_encoder(name: str) -> Any:
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        # Avoid hard-failing on environments with restricted outbound network or missing caches.
        # This keeps providers functional and preserves ability to estimate budgets.
        logger.warning("tiktoken encoding load failed; falling back to approximate tokenizer", extra={"encoding": name})
        return _FallbackEncoding(name)


class ModelProfile:
    supported_models = {
        "completions": [
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.2",
            "gpt-5.2-chat-latest",
            "gpt-5.2-pro",
            "gpt-5-pro",
            "gpt-5-codex",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "chatgpt-4o-latest",
            "gpt-4o-mini",
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
            "computer-use-preview",
            "gpt-oss-120b",
            "gpt-oss-20b",
            "o3",
            "o3-mini",
            "o3-pro",
            "o4-mini",
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-pro",
        ],
        "embeddings": [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ],
        "images": [
            "chatgpt-image-latest",
            "gpt-image-1.5",
            "gpt-image-1",
            "gpt-image-1-mini",
            "dall-e-2",
            "dall-e-3",
        ],
        "audio": [
            "gpt-audio",
            "gpt-audio-mini",
            "gpt-4o-audio-preview",
            "gpt-4o-mini-audio-preview",
            "whisper-1",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe-diarize",
            "gpt-4o-mini-tts",
            "tts-1",
            "tts-1-hd",
        ],
        "moderations": [
            "omni-moderation-latest",
            "text-moderation-latest",
            "text-moderation-stable",
        ],
        "realtime": [
            "gpt-realtime",
            "gpt-realtime-mini",
            "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview",
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
    deprecated: ClassVar[bool] = False
    replacement: ClassVar[str | None] = None
    structured_outputs_support: ClassVar[bool | None] = None
    responses_api_support: ClassVar[bool | None] = None
    background_responses_support: ClassVar[bool | None] = None
    responses_native_tools_support: ClassVar[bool | None] = None
    normalized_output_items_support: ClassVar[bool | None] = None
    vision_input_support: ClassVar[bool | None] = None
    audio_input_support: ClassVar[bool | None] = None
    file_input_support: ClassVar[bool | None] = None

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
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        parsed_usage: dict[str, int | Decimal] = {
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
        }

        output_tokens: int | None = None
        if "completion_tokens" in usage:
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            parsed_usage["output_tokens"] = output_tokens
        elif "output_tokens" in usage:
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            parsed_usage["output_tokens"] = output_tokens

        cached_tokens = 0
        prompt_tokens_details = usage.get("prompt_tokens_details")
        input_tokens_details = usage.get("input_tokens_details")
        token_details = prompt_tokens_details if isinstance(prompt_tokens_details, dict) else input_tokens_details
        if isinstance(token_details, dict):
            cached_tokens = int(token_details.get("cached_tokens", 0) or 0)

        output_tokens_reasoning = 0
        output_tokens_details = usage.get("output_tokens_details")
        if isinstance(output_tokens_details, dict):
            output_tokens_reasoning = int(output_tokens_details.get("reasoning_tokens", 0) or 0)
            if output_tokens_reasoning > 0:
                parsed_usage["output_tokens_reasoning"] = output_tokens_reasoning

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
            if key.startswith(("ft:", "ft-")):
                base_key = "gpt-4o-mini"
                if key.startswith("ft:"):
                    parts = key.split(":")
                    if len(parts) > 1 and parts[1]:
                        base_key = parts[1]
                try:
                    base_profile = cls.get(base_key)
                except ValueError:
                    base_profile = GPT5Mini
                dynamic_name = "DynamicFineTuned_" + re.sub(r"[^0-9A-Za-z_]", "_", key)
                return type(
                    dynamic_name,
                    (cls,),
                    {
                        "key": key,
                        "model_name": key,
                        "category": base_profile.category,
                        "context_window": base_profile.context_window,
                        "max_output": base_profile.max_output,
                        "output_dimensions": base_profile.output_dimensions,
                        "usage_costs": dict(base_profile.usage_costs),
                        "rate_limits": dict(base_profile.rate_limits),
                        "reasoning_model": base_profile.reasoning_model,
                        "reasoning_efforts": list(base_profile.reasoning_efforts),
                        "default_reasoning_effort": base_profile.default_reasoning_effort,
                        "function_calling_support": base_profile.function_calling_support,
                        "token_streaming_support": base_profile.token_streaming_support,
                        "encoding": base_profile.encoding,
                    },
                )
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


class GPT5Pro(ModelProfile):
    key = "gpt-5-pro"
    model_name = "gpt-5-pro-2025-12-11"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("15.00") / Decimal("1000000"),
        "output": Decimal("120.00") / Decimal("1000000"),
        "cached_input": Decimal("1.50") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 500_000, "req_per_min": 5_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5Codex(ModelProfile):
    key = "gpt-5-codex"
    model_name = "gpt-5-codex-2025-12-11"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("1.50") / Decimal("1000000"),
        "output": Decimal("12.00") / Decimal("1000000"),
        "cached_input": Decimal("0.15") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT52Pro(ModelProfile):
    key = "gpt-5.2-pro"
    model_name = "gpt-5.2-pro-2025-12-11"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("18.00") / Decimal("1000000"),
        "output": Decimal("144.00") / Decimal("1000000"),
        "cached_input": Decimal("1.80") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 500_000, "req_per_min": 3_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT52ChatLatest(ModelProfile):
    key = "gpt-5.2-chat-latest"
    model_name = "gpt-5.2-chat-latest"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = dict(GPT5Point2.usage_costs)
    rate_limits = dict(GPT5Point2.rate_limits)
    reasoning_model = True
    reasoning_efforts = list(GPT5Point2.reasoning_efforts)
    default_reasoning_effort = GPT5Point2.default_reasoning_effort
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT51Codex(ModelProfile):
    key = "gpt-5.1-codex"
    model_name = "gpt-5.1-codex-2025-11-13"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("1.40") / Decimal("1000000"),
        "output": Decimal("11.20") / Decimal("1000000"),
        "cached_input": Decimal("0.14") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 1_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["none", "minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT51CodexMax(ModelProfile):
    key = "gpt-5.1-codex-max"
    model_name = "gpt-5.1-codex-max-2025-11-13"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("2.10") / Decimal("1000000"),
        "output": Decimal("16.80") / Decimal("1000000"),
        "cached_input": Decimal("0.21") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 750_000, "req_per_min": 8_000}
    reasoning_model = True
    reasoning_efforts = ["none", "minimal", "low", "medium", "high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT51CodexMini(ModelProfile):
    key = "gpt-5.1-codex-mini"
    model_name = "gpt-5.1-codex-mini-2025-11-13"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {
        "input": Decimal("0.40") / Decimal("1000000"),
        "output": Decimal("3.20") / Decimal("1000000"),
        "cached_input": Decimal("0.04") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 4_000_000, "req_per_min": 10_000}
    reasoning_model = True
    reasoning_efforts = ["none", "minimal", "low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT51ChatLatest(ModelProfile):
    key = "gpt-5.1-chat-latest"
    model_name = "gpt-5.1-chat-latest"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = dict(GPT5Point1.usage_costs)
    rate_limits = dict(GPT5Point1.rate_limits)
    reasoning_model = True
    reasoning_efforts = list(GPT5Point1.reasoning_efforts)
    default_reasoning_effort = GPT5Point1.default_reasoning_effort
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT5ChatLatest(ModelProfile):
    key = "gpt-5-chat-latest"
    model_name = "gpt-5-chat-latest"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = dict(GPT5.usage_costs)
    rate_limits = dict(GPT5.rate_limits)
    reasoning_model = True
    reasoning_efforts = list(GPT5.reasoning_efforts)
    default_reasoning_effort = GPT5.default_reasoning_effort
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT41(ModelProfile):
    key = "gpt-4.1"
    model_name = "gpt-4.1-2025-04-14"
    category = "completions"
    context_window = 1_000_000
    max_output = 32_768
    usage_costs = {
        "input": Decimal("2.00") / Decimal("1000000"),
        "output": Decimal("8.00") / Decimal("1000000"),
        "cached_input": Decimal("0.50") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT41Mini(ModelProfile):
    key = "gpt-4.1-mini"
    model_name = "gpt-4.1-mini-2025-04-14"
    category = "completions"
    context_window = 1_000_000
    max_output = 32_768
    usage_costs = {
        "input": Decimal("0.40") / Decimal("1000000"),
        "output": Decimal("1.60") / Decimal("1000000"),
        "cached_input": Decimal("0.10") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 6_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT41Nano(ModelProfile):
    key = "gpt-4.1-nano"
    model_name = "gpt-4.1-nano-2025-04-14"
    category = "completions"
    context_window = 1_000_000
    max_output = 32_768
    usage_costs = {
        "input": Decimal("0.10") / Decimal("1000000"),
        "output": Decimal("0.40") / Decimal("1000000"),
        "cached_input": Decimal("0.025") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    responses_native_tools_support = False


class GPT4o(ModelProfile):
    key = "gpt-4o"
    model_name = "gpt-4o-2024-08-06"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = {
        "input": Decimal("2.50") / Decimal("1000000"),
        "output": Decimal("10.00") / Decimal("1000000"),
        "cached_input": Decimal("1.25") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 2_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT4oMini(ModelProfile):
    key = "gpt-4o-mini"
    model_name = "gpt-4o-mini-2024-07-18"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = {
        "input": Decimal("0.15") / Decimal("1000000"),
        "output": Decimal("0.60") / Decimal("1000000"),
        "cached_input": Decimal("0.075") / Decimal("1000000"),
    }
    rate_limits = {"tkn_per_min": 10_000_000, "req_per_min": 10_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class ChatGPT4oLatest(ModelProfile):
    key = "chatgpt-4o-latest"
    model_name = "chatgpt-4o-latest"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = dict(GPT4o.usage_costs)
    rate_limits = dict(GPT4o.rate_limits)
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT4oSearchPreview(ModelProfile):
    key = "gpt-4o-search-preview"
    model_name = "gpt-4o-search-preview"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = dict(GPT4o.usage_costs)
    rate_limits = dict(GPT4o.rate_limits)
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    responses_api_support = False
    background_responses_support = False
    responses_native_tools_support = False
    normalized_output_items_support = False


class GPT4oMiniSearchPreview(ModelProfile):
    key = "gpt-4o-mini-search-preview"
    model_name = "gpt-4o-mini-search-preview"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = dict(GPT4oMini.usage_costs)
    rate_limits = dict(GPT4oMini.rate_limits)
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    responses_api_support = False
    background_responses_support = False
    responses_native_tools_support = False
    normalized_output_items_support = False


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


class TextEmbeddingAda002(ModelProfile):
    key = "text-embedding-ada-002"
    model_name = "text-embedding-ada-002"
    category = "embeddings"
    context_window = 8_191
    output_dimensions = 1_536
    usage_costs = {"input": Decimal("0.10") / Decimal("1000000")}
    rate_limits = {"tkn_per_min": 500_000, "req_per_min": 3_000}
    encoding = "cl100k_base"
    deprecated = True
    replacement = "text-embedding-3-small"


class GPTImage15(ModelProfile):
    key = "gpt-image-1.5"
    model_name = "gpt-image-1.5"
    category = "images"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"


class ChatGPTImageLatest(ModelProfile):
    key = "chatgpt-image-latest"
    model_name = "chatgpt-image-latest"
    category = "images"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"


class GPTImage1(ModelProfile):
    key = "gpt-image-1"
    model_name = "gpt-image-1"
    category = "images"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"


class GPTImage1Mini(ModelProfile):
    key = "gpt-image-1-mini"
    model_name = "gpt-image-1-mini"
    category = "images"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"


class DallE2(ModelProfile):
    key = "dall-e-2"
    model_name = "dall-e-2"
    category = "images"
    context_window = 8_192
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 200}
    encoding = "cl100k_base"
    deprecated = True
    replacement = "gpt-image-1"
    vision_input_support = False
    audio_input_support = False
    file_input_support = False


class DallE3(ModelProfile):
    key = "dall-e-3"
    model_name = "dall-e-3"
    category = "images"
    context_window = 8_192
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 200}
    encoding = "cl100k_base"
    deprecated = True
    replacement = "gpt-image-1"
    vision_input_support = False
    audio_input_support = False
    file_input_support = False


class GPTAudio(ModelProfile):
    key = "gpt-audio"
    model_name = "gpt-audio"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class GPTAudioMini(ModelProfile):
    key = "gpt-audio-mini"
    model_name = "gpt-audio-mini"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class GPT4oAudioPreview(ModelProfile):
    key = "gpt-4o-audio-preview"
    model_name = "gpt-4o-audio-preview"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-audio"


class GPT4oMiniAudioPreview(ModelProfile):
    key = "gpt-4o-mini-audio-preview"
    model_name = "gpt-4o-mini-audio-preview"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-audio-mini"


class Whisper1(ModelProfile):
    key = "whisper-1"
    model_name = "whisper-1"
    category = "audio"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "cl100k_base"


class GPT4oTranscribe(ModelProfile):
    key = "gpt-4o-transcribe"
    model_name = "gpt-4o-transcribe"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class GPT4oMiniTranscribe(ModelProfile):
    key = "gpt-4o-mini-transcribe"
    model_name = "gpt-4o-mini-transcribe"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class GPT4oTranscribeDiarize(ModelProfile):
    key = "gpt-4o-transcribe-diarize"
    model_name = "gpt-4o-transcribe-diarize"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class GPT4oMiniTTS(ModelProfile):
    key = "gpt-4o-mini-tts"
    model_name = "gpt-4o-mini-tts"
    category = "audio"
    context_window = 128_000
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class TTS1(ModelProfile):
    key = "tts-1"
    model_name = "tts-1"
    category = "audio"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "cl100k_base"


class TTS1HD(ModelProfile):
    key = "tts-1-hd"
    model_name = "tts-1-hd"
    category = "audio"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 500}
    encoding = "cl100k_base"


class OmniModerationLatest(ModelProfile):
    key = "omni-moderation-latest"
    model_name = "omni-moderation-latest"
    category = "moderations"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "o200k_base"


class TextModerationLatest(ModelProfile):
    key = "text-moderation-latest"
    model_name = "text-moderation-latest"
    category = "moderations"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "cl100k_base"


class TextModerationStable(ModelProfile):
    key = "text-moderation-stable"
    model_name = "text-moderation-stable"
    category = "moderations"
    context_window = 32_768
    usage_costs = {"input": Decimal("0.00"), "output": Decimal("0.00"), "cached_input": Decimal("0.00")}
    rate_limits = {"req_per_min": 1_000}
    encoding = "cl100k_base"


class GPTRealtime(ModelProfile):
    key = "gpt-realtime"
    model_name = "gpt-realtime"
    category = "realtime"
    context_window = 128_000
    max_output = 4_096
    usage_costs = {"input": Decimal("4.00") / Decimal("1000000"), "output": Decimal("16.00") / Decimal("1000000"), "cached_input": Decimal("0.40") / Decimal("1000000")}
    rate_limits = {"req_per_min": 500, "tkn_per_min": 1_000_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPTRealtimeMini(ModelProfile):
    key = "gpt-realtime-mini"
    model_name = "gpt-realtime-mini"
    category = "realtime"
    context_window = 128_000
    max_output = 4_096
    usage_costs = {"input": Decimal("0.60") / Decimal("1000000"), "output": Decimal("2.40") / Decimal("1000000"), "cached_input": Decimal("0.06") / Decimal("1000000")}
    rate_limits = {"req_per_min": 1_000, "tkn_per_min": 2_000_000}
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class GPT4oRealtimePreview(ModelProfile):
    key = "gpt-4o-realtime-preview"
    model_name = "gpt-4o-realtime-preview"
    category = "realtime"
    context_window = 128_000
    max_output = 4_096
    usage_costs = dict(GPTRealtime.usage_costs)
    rate_limits = dict(GPTRealtime.rate_limits)
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-realtime"


class GPT4oMiniRealtimePreview(ModelProfile):
    key = "gpt-4o-mini-realtime-preview"
    model_name = "gpt-4o-mini-realtime-preview"
    category = "realtime"
    context_window = 128_000
    max_output = 4_096
    usage_costs = dict(GPTRealtimeMini.usage_costs)
    rate_limits = dict(GPTRealtimeMini.rate_limits)
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-realtime-mini"


class ComputerUsePreview(ModelProfile):
    key = "computer-use-preview"
    model_name = "computer-use-preview"
    category = "completions"
    context_window = 128_000
    max_output = 16_384
    usage_costs = dict(GPT4o.usage_costs)
    rate_limits = {"tkn_per_min": 500_000, "req_per_min": 2_000}
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"
    structured_outputs_support = False
    responses_native_tools_support = True
    normalized_output_items_support = True


class O3(ModelProfile):
    key = "o3"
    model_name = "o3-2025-04-16"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("8.00") / Decimal("1000000"),
        "output": Decimal("32.00") / Decimal("1000000"),
        "cached_input": Decimal("0.80") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 1_000, "tkn_per_min": 1_000_000}
    reasoning_model = True
    reasoning_efforts = ["medium", "high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class O3Mini(ModelProfile):
    key = "o3-mini"
    model_name = "o3-mini-2025-01-31"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("1.50") / Decimal("1000000"),
        "output": Decimal("6.00") / Decimal("1000000"),
        "cached_input": Decimal("0.15") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 3_000, "tkn_per_min": 2_000_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-5-mini"


class O3Pro(ModelProfile):
    key = "o3-pro"
    model_name = "o3-pro-2025-06-10"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("12.00") / Decimal("1000000"),
        "output": Decimal("48.00") / Decimal("1000000"),
        "cached_input": Decimal("1.20") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 500, "tkn_per_min": 500_000}
    reasoning_model = True
    reasoning_efforts = ["high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class O4Mini(ModelProfile):
    key = "o4-mini"
    model_name = "o4-mini-2025-04-16"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("12.00") / Decimal("1000000"),
        "cached_input": Decimal("0.30") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 2_000, "tkn_per_min": 2_000_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"


class O1(ModelProfile):
    key = "o1"
    model_name = "o1-2024-12-17"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("15.00") / Decimal("1000000"),
        "output": Decimal("60.00") / Decimal("1000000"),
        "cached_input": Decimal("1.50") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 500, "tkn_per_min": 500_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-5"


class O1Mini(ModelProfile):
    key = "o1-mini"
    model_name = "o1-mini-2024-09-12"
    category = "completions"
    context_window = 128_000
    max_output = 65_536
    usage_costs = {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("12.00") / Decimal("1000000"),
        "cached_input": Decimal("0.30") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 1_000, "tkn_per_min": 1_000_000}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-5-mini"


class O1Preview(ModelProfile):
    key = "o1-preview"
    model_name = "o1-preview-2024-09-12"
    category = "completions"
    context_window = 128_000
    max_output = 65_536
    usage_costs = {
        "input": Decimal("15.00") / Decimal("1000000"),
        "output": Decimal("60.00") / Decimal("1000000"),
        "cached_input": Decimal("1.50") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 250, "tkn_per_min": 250_000}
    reasoning_model = True
    reasoning_efforts = ["medium", "high"]
    default_reasoning_effort = "high"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-5"


class O1Pro(ModelProfile):
    key = "o1-pro"
    model_name = "o1-pro-2025-03-19"
    category = "completions"
    context_window = 200_000
    max_output = 100_000
    usage_costs = {
        "input": Decimal("20.00") / Decimal("1000000"),
        "output": Decimal("80.00") / Decimal("1000000"),
        "cached_input": Decimal("2.00") / Decimal("1000000"),
    }
    rate_limits = {"req_per_min": 250, "tkn_per_min": 250_000}
    reasoning_model = True
    reasoning_efforts = ["high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"
    deprecated = True
    replacement = "gpt-5-pro"


class O3DeepResearch(ModelProfile):
    key = "o3-deep-research"
    model_name = "o3-deep-research"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {"input": Decimal("10.00") / Decimal("1000000"), "output": Decimal("40.00") / Decimal("1000000"), "cached_input": Decimal("1.00") / Decimal("1000000")}
    rate_limits = {"req_per_min": 500, "tkn_per_min": 500_000}
    reasoning_model = True
    reasoning_efforts = ["high", "xhigh"]
    default_reasoning_effort = "high"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"


class O4MiniDeepResearch(ModelProfile):
    key = "o4-mini-deep-research"
    model_name = "o4-mini-deep-research"
    category = "completions"
    context_window = 400_000
    max_output = 128_000
    usage_costs = {"input": Decimal("2.00") / Decimal("1000000"), "output": Decimal("8.00") / Decimal("1000000"), "cached_input": Decimal("0.20") / Decimal("1000000")}
    rate_limits = {"req_per_min": 1_000, "tkn_per_min": 1_000_000}
    reasoning_model = True
    reasoning_efforts = ["high"]
    default_reasoning_effort = "high"
    function_calling_support = False
    token_streaming_support = True
    encoding = "o200k_base"


class GPTOSS120B(ModelProfile):
    key = "gpt-oss-120b"
    model_name = "gpt-oss-120b"
    category = "completions"
    context_window = 131_072
    max_output = 16_384
    usage_costs = {
        "input": Decimal("0.00"),
        "output": Decimal("0.00"),
        "cached_input": Decimal("0.00"),
    }
    rate_limits = {"req_per_min": 0, "tkn_per_min": 0}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    responses_api_support = False
    background_responses_support = False
    responses_native_tools_support = False
    normalized_output_items_support = False
    audio_input_support = False


class GPTOSS20B(ModelProfile):
    key = "gpt-oss-20b"
    model_name = "gpt-oss-20b"
    category = "completions"
    context_window = 131_072
    max_output = 16_384
    usage_costs = {
        "input": Decimal("0.00"),
        "output": Decimal("0.00"),
        "cached_input": Decimal("0.00"),
    }
    rate_limits = {"req_per_min": 0, "tkn_per_min": 0}
    reasoning_model = True
    reasoning_efforts = ["low", "medium", "high"]
    default_reasoning_effort = "medium"
    function_calling_support = True
    token_streaming_support = True
    encoding = "o200k_base"
    responses_api_support = False
    background_responses_support = False
    responses_native_tools_support = False
    normalized_output_items_support = False
    audio_input_support = False


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


class ClaudeSonnet4(ModelProfile):
    """Claude Sonnet 4 - current Anthropic Sonnet family entry."""

    key = "claude-sonnet-4"
    model_name = "claude-sonnet-4-20250514"
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


class ClaudeOpus4(ModelProfile):
    """Claude Opus 4 - current Anthropic Opus family entry."""

    key = "claude-opus-4"
    model_name = "claude-opus-4-20250514"
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


class Claude45Haiku(ModelProfile):
    """Compatibility alias for the latest supported Haiku-class Anthropic model."""

    key = "claude-4-5-haiku"
    model_name = "claude-3-5-haiku-20241022"
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
    """Compatibility alias for the current Claude Sonnet 4 model."""

    key = "claude-4-5-sonnet"
    model_name = "claude-sonnet-4-20250514"
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
    """Compatibility alias for the current Claude Opus 4 model."""

    key = "claude-4-5-opus"
    model_name = "claude-opus-4-20250514"
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
    "GPT5ChatLatest",
    "GPT5Mini",
    "GPT5Nano",
    "GPT5Pro",
    "GPT5Codex",
    "GPT5Point1",
    "GPT5Point2",
    "GPT52Pro",
    "GPT52ChatLatest",
    "GPT51Codex",
    "GPT51CodexMax",
    "GPT51CodexMini",
    "GPT51ChatLatest",
    "GPT41",
    "GPT41Mini",
    "GPT41Nano",
    "GPT4o",
    "ChatGPT4oLatest",
    "GPT4oMini",
    "GPT4oSearchPreview",
    "GPT4oMiniSearchPreview",
    "TextEmbedding3Large",
    "TextEmbedding3Small",
    "TextEmbeddingAda002",
    "GPTImage15",
    "ChatGPTImageLatest",
    "GPTImage1",
    "GPTImage1Mini",
    "DallE2",
    "DallE3",
    "GPTAudio",
    "GPTAudioMini",
    "GPT4oAudioPreview",
    "GPT4oMiniAudioPreview",
    "Whisper1",
    "GPT4oTranscribe",
    "GPT4oMiniTranscribe",
    "GPT4oTranscribeDiarize",
    "GPT4oMiniTTS",
    "TTS1",
    "TTS1HD",
    "OmniModerationLatest",
    "TextModerationLatest",
    "TextModerationStable",
    "GPTRealtime",
    "GPTRealtimeMini",
    "GPT4oRealtimePreview",
    "GPT4oMiniRealtimePreview",
    "ComputerUsePreview",
    "O3",
    "O3Mini",
    "O3Pro",
    "O4Mini",
    "O1",
    "O1Mini",
    "O1Preview",
    "O1Pro",
    "O3DeepResearch",
    "O4MiniDeepResearch",
    "GPTOSS120B",
    "GPTOSS20B",
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
