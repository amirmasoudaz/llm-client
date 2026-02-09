# src/agents/detailed_generation/agent.py

import logging

from blake3 import blake3
from llm_client import OpenAIClient, GPT5Point2

from src.agents.detailed_generation.prompt import SYSTEM_PROMPT, PROMPT_VERSION
from src.agents.detailed_generation.schema import RespSchema
from src.agents.utils import safe_llm_call

logger = logging.getLogger(__name__)


class DetailedEmailGenerationAgent:
    collection = "funding_detailed_email_generation_agent_logs"

    def __init__(self):
        self.openai_client = OpenAIClient(
            model=GPT5Point2,
            cache_backend="pg_redis",
            cache_collection=self.collection
        )

    async def generate(
            self,
            funding_request_id: int,
            research_interest: str,
            paper_title: str,
            paper_journal: str,
            paper_year: int,
            research_connection: str,
            professor_last_name: str,
            template_body: str,
            regen: bool = False,
            raise_on_failure: bool = False,
        ) -> str | None:
        """
        Generate a detailed email with robust error handling.
        
        Args:
            funding_request_id: Unique ID for the funding request
            research_interest: Professor's research interests
            paper_title: Title of the relevant paper
            paper_journal: Journal where the paper was published
            paper_year: Year the paper was published
            research_connection: Connection to the research in the paper
            professor_last_name: Last name of the professor
            template_body: Email template body with placeholders
            regen: Whether to regenerate cached response
            raise_on_failure: If True, raise exception on failure; else return None
            
        Returns:
            Generated email body string, or None on failure
        """
        email_body = template_body.replace("{{ProfessorName}}", professor_last_name)
        email_body = email_body.replace("{{ProfessorInterests}}", research_interest)
        email_body = email_body.replace("{{PaperTitle}}", paper_title)
        if paper_year is not None:
            email_body = email_body.replace("{{Year}}", str(paper_year))
        if paper_journal is not None:
            email_body = email_body.replace("{{JournalName}}", paper_journal)
        email_body = email_body.replace("{{ResearchConnection}}", "{{RESEARCH_CONNECTION}}")  # placeholder for generation

        content = "\n\n".join([
            "BASE_EMAIL: ```\n" + email_body + "\n```\n\n",
            "INTERESTS: ###\n" + research_interest + "\n###\n\n",
            "PAPER_TITLE: **" + paper_title + "**\n\n",
            "ABSTRACT: @@@\n" + research_connection + "\n@@@"
        ])
        content_hash = blake3(content.encode('utf-8')).hexdigest()
        identifier = f"generation_{funding_request_id}_{content_hash}_{PROMPT_VERSION}"

        async def _make_llm_call():
            return await self.openai_client.get_response(
                identifier=identifier,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                response_format=RespSchema,
                reasoning_effort="medium",
                cache_response=True,
                regen_cache=regen,
                timeout=30,
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
            logger.warning(f"[{identifier}] LLM call failed, returning template body as fallback")
            return email_body  # Return the template with placeholders as fallback

        output = result.get("output", {})
        if not output or "email" not in output:
            logger.warning(f"[{identifier}] Empty output, returning template body as fallback")
            return email_body

        return output["email"]
