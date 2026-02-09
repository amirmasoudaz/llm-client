import asyncio
import json
import logging
from typing import List, Any

from blake3 import blake3
from tqdm import tqdm
from llm_client import OpenAIClient, GPT5Nano

from src.config import settings
from src.agents.tag_generation.prompt import SYSTEM_PROMPT, PROMPT_VERSION
from src.agents.tag_generation.schema import RespSchema
from src.agents.utils import safe_llm_call, get_fallback_tags

logger = logging.getLogger(__name__)


class TagGenerationAgent:
    collection = "funding_recommendation_tag_generation_agent_logs"

    def __init__(self):
        self.openai_client = OpenAIClient(
            model=GPT5Nano,
            cache_backend="pg_redis",
            cache_collection=self.collection
        )

    @staticmethod
    def _build_user_prompt(row: dict[str, Any]) -> str:
        blob = {
            "full_name": row.get("full_name", ""),
            "department": row.get("department", ""),
            "institute": row.get("institute", ""),
            "occupation": row.get("occupation", ""),
            "research_areas": row.get("research_areas") or [],
            "area_of_expertise": row.get("area_of_expertise") or [],
            "credentials": row.get("credentials", ""),
            "url": row.get("url"),
        }
        return json.dumps(blob, ensure_ascii=False)

    async def batch_generate(self, contents: List[dict], regen: bool = False):
        """
        Generate tags for a batch of professor profiles with robust error handling.
        
        Args:
            contents: List of professor profile dicts
            regen: Whether to regenerate cached responses
            
        Returns:
            List of (row, tags) tuples for successful generations
        """
        if not contents:
            return []

        await self.openai_client.warm_cache()

        tasks = [self.generate(row, regen=regen) for row in contents]
        progress = tqdm(total=len(tasks), desc="Tagging batches")
        outputs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        failed_count = 0

        for i in range(0, len(tasks), settings.TAGGER_MAX_CONCURRENCY):
            chunk = tasks[i: i + settings.TAGGER_MAX_CONCURRENCY]
            results = await asyncio.gather(*chunk, return_exceptions=True)
            
            for j, res in enumerate(results):
                row_idx = i + j
                row_obj = contents[row_idx] if row_idx < len(contents) else {}
                
                if isinstance(res, Exception):
                    logger.warning(f"Tag generation failed for row {row_idx}: {res}")
                    failed_count += 1
                    continue
                    
                if isinstance(res, dict):
                    # Check if it's a fallback response
                    if res.get("error") == "LLM_FAILED":
                        failed_count += 1
                        continue
                    outputs.append((row_obj, res))
                    
            progress.update(len(chunk))

        if progress:
            progress.close()

        if failed_count > 0:
            logger.warning(f"Tag generation: {failed_count} failures out of {len(contents)} total")

        return outputs

    async def generate(
        self, 
        content: dict[str, Any], 
        regen: bool = False,
        raise_on_failure: bool = False,
    ) -> dict[str, Any]:
        """
        Generate tags for a professor profile with robust error handling.
        
        Args:
            content: Professor profile dict
            regen: Whether to regenerate cached response
            raise_on_failure: If True, raise exception on failure; else return fallback
            
        Returns:
            Tags dict with generated tags, or fallback on failure
        """
        tag_context = self._build_user_prompt(content)

        prof_hash = content.get("prof_hash", None)
        full_name = content.get("full_name", "")
        institute = content.get("institute", "")

        ident = f"{full_name}|{institute}|{prof_hash}"
        ident_hash = blake3(ident.encode("utf-8")).hexdigest()
        identifier = f"tag_generation_{ident_hash}_{PROMPT_VERSION}"

        async def _make_llm_call():
            return await self.openai_client.get_response(
                identifier=identifier,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tag_context}
                ],
                response_format=RespSchema,
                reasoning_effort="minimal",
                cache_response=True,
                regen_cache=regen,
                timeout=45,
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
            logger.warning(f"[{identifier}] LLM call failed, using fallback tags")
            fallback = get_fallback_tags()
            fallback.update({
                "prof_hash": prof_hash,
                "full_name": full_name,
                "institute": institute
            })
            return fallback

        output = result.get("output", {})
        if not output:
            logger.warning(f"[{identifier}] Empty output, using fallback tags")
            fallback = get_fallback_tags()
            fallback.update({
                "prof_hash": prof_hash,
                "full_name": full_name,
                "institute": institute
            })
            return fallback

        tags = output

        if isinstance(prof_hash, (bytearray, memoryview)):
            prof_hash = bytes(prof_hash)

        tags.update({
            "prof_hash": prof_hash,
            "full_name": full_name,
            "institute": institute
        })

        return tags
