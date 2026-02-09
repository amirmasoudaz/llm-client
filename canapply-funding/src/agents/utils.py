"""
Shared utilities for agent error handling, retries, and fallbacks.
"""
import asyncio
import logging
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class LLMResponseError(AgentError):
    """Raised when LLM returns None or invalid response."""
    pass


class MaxRetriesExceeded(AgentError):
    """Raised when all retry attempts are exhausted."""
    pass


async def safe_llm_call(
    llm_call: Callable[[], Any],
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    fallback: Any = None,
    identifier: str = "unknown",
    raise_on_failure: bool = False,
) -> tuple[Any, bool]:
    """
    Safely execute an LLM call with retries and fallback.
    
    Args:
        llm_call: Async callable that makes the LLM request
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
        fallback: Value to return if all retries fail
        identifier: Identifier for logging purposes
        raise_on_failure: If True, raise exception instead of returning fallback
        
    Returns:
        Tuple of (result, success_bool)
    """
    last_exception = None
    delay = retry_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await llm_call()
            
            # Validate response
            if result is None:
                raise LLMResponseError(f"LLM returned None for {identifier}")
            
            if isinstance(result, dict):
                if result.get("error") and result.get("error") != "OK":
                    raise LLMResponseError(f"LLM error for {identifier}: {result.get('error')}")
                if "output" in result and result["output"] is None:
                    raise LLMResponseError(f"LLM output is None for {identifier}")
            
            return result, True
            
        except Exception as e:
            last_exception = e
            logger.warning(
                f"[{identifier}] Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}"
            )
            
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff_multiplier
    
    # All retries exhausted
    logger.error(f"[{identifier}] All {max_retries} attempts failed. Last error: {last_exception}")
    
    if raise_on_failure:
        raise MaxRetriesExceeded(
            f"All {max_retries} attempts failed for {identifier}: {last_exception}"
        ) from last_exception
    
    return fallback, False


def get_fallback_digestion(funding_request_id: int, professor_reply_body: str) -> dict:
    """Return a safe fallback response for reply digestion."""
    return {
        "funding_request_id": funding_request_id,
        "reply_body_raw": professor_reply_body,
        "reply_body_cleaned": professor_reply_body,
        "is_auto_generated": False,
        "auto_generated_type": "NONE",
        "needs_human_review": True,  # Flag for manual review
        "engagement_label": "AMBIGUOUS_OR_UNCLEAR",
        "engagement_bool": None,
        "activity_status": "UNKNOWN",
        "activity_bool": None,
        "next_step_type": "NO_NEXT_STEP",
        "short_rationale": "LLM call failed - requires human review",
        "key_phrases": [],
        "confidence": 0.0,
    }


def get_fallback_tags() -> dict:
    """Return a safe fallback response for tag generation."""
    return {
        "tags": [],
        "primary_field": "Unknown",
        "secondary_fields": [],
        "error": "LLM_FAILED",
    }
