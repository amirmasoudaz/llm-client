import logging

from blake3 import blake3
from llm_client import OpenAIClient, GPT5Mini

from src.agents.email_paraphrasing.prompt import SYSTEM_PROMPT, PROMPT_VERSION
from src.agents.email_paraphrasing.schema import RespSchema
from src.agents.utils import safe_llm_call

logger = logging.getLogger(__name__)


class ParaphrasingAgent:
    collection = "funding_email_paraphrasing_agent_logs"

    def __init__(self):
        self.openai_client = OpenAIClient(
            model=GPT5Mini,
            cache_backend="pg_redis",
            cache_collection=self.collection
        )

    async def paraphrase(
            self,
            content: str,
            identifier: str = None,
            regen: bool = False,
            raise_on_failure: bool = False,
        ) -> str:
        """
        Paraphrase email content with robust error handling.
        
        Args:
            content: The content to paraphrase
            identifier: Optional cache identifier
            regen: Whether to regenerate cached response
            raise_on_failure: If True, raise exception on failure; else return original
            
        Returns:
            Paraphrased email body string, or original content on failure
        """
        if identifier is None:
            content_hash = blake3(content.encode('utf-8')).hexdigest()
            identifier = f"paraphrasing_{content_hash}"
        assert identifier is not None and isinstance(identifier, str), "Identifier must be a non-empty string."
        identifier += f"_{PROMPT_VERSION}"

        async def _make_llm_call():
            return await self.openai_client.get_response(
                identifier=identifier,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                response_format=RespSchema,
                reasoning_effort="low",
                cache_response=True,
                regen_cache=regen,
                timeout=30,
                attempts=1
            )

        # Execute with retries and fallback
        result, success = await safe_llm_call(
            _make_llm_call,
            max_retries=3,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            fallback=None,
            identifier=identifier,
            raise_on_failure=raise_on_failure,
        )

        if not success or result is None:
            logger.warning(f"[{identifier}] LLM call failed, returning original content")
            # Extract just the email body from content if possible
            if "Raw Email:" in content:
                return content.split("Raw Email:")[-1].strip()
            return content

        output = result.get("output", {})
        if not output or "email" not in output:
            logger.warning(f"[{identifier}] Empty output, returning original content")
            if "Raw Email:" in content:
                return content.split("Raw Email:")[-1].strip()
            return content

        return output["email"].strip()
