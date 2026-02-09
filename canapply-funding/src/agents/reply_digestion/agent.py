import asyncio
import json
import logging

from blake3 import blake3
from llm_client import OpenAIClient, GPT5Nano

from src.agents.reply_digestion.prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT
from src.agents.reply_digestion.schemas import RESPONSE_SCHEMA
from src.agents.utils import safe_llm_call, get_fallback_digestion

from src.db.session import DB
from src.tools.helpers import ext_student_name_subject

logger = logging.getLogger(__name__)


class ReplyDigestionAgent:
    collection = "funding_reply_digestion_agent_logs"

    def __init__(self):
        self.openai_client = OpenAIClient(
            model=GPT5Nano,
            cache_backend="pg_redis",
            cache_collection=self.collection
        )

    async def digest(
        self, 
        funding_request_id: int,
        professor_reply_body: str, 
        main_email_body: str,
        main_email_subject: str,
        professor_name: str,
        regen: bool = False,
        raise_on_failure: bool = False,
    ) -> dict:
        """
        Digest a professor reply with robust error handling.
        
        Args:
            funding_request_id: ID of the funding request
            professor_reply_body: The professor's reply email body
            main_email_body: The original student email body
            main_email_subject: The original email subject
            professor_name: Name of the professor
            regen: Whether to regenerate cached response
            raise_on_failure: If True, raise exception on failure; else return fallback
            
        Returns:
            Digested reply dict with all classification fields
        """
        # Build unified context with all inputs
        input_payload = {
            "student_email": main_email_body,
            "professor_reply": professor_reply_body,
            "professor_name": professor_name,
            "student_name": ext_student_name_subject(main_email_subject)
        }
        input_context = json.dumps(input_payload, ensure_ascii=False)
        
        # Generate cache key
        context_hash = blake3(input_context.encode('utf-8')).hexdigest()
        identifier = f"reply_digestion_{funding_request_id}_{context_hash}_{PROMPT_VERSION}"

        async def _make_llm_call():
            return await self.openai_client.get_response(
                identifier=identifier,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(input_context=input_context)}
                ],
                response_format=RESPONSE_SCHEMA,
                reasoning_effort="minimal",
                cache_response=True,
                regen_cache=regen,
                timeout=30,
                attempts=1  # Internal retries handled by safe_llm_call
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
            logger.warning(f"[{identifier}] Using fallback response")
            return get_fallback_digestion(funding_request_id, professor_reply_body)

        output = result.get("output", {})
        if not output:
            logger.warning(f"[{identifier}] Empty output, using fallback")
            return get_fallback_digestion(funding_request_id, professor_reply_body)

        output["funding_request_id"] = funding_request_id
        output["reply_body_raw"] = professor_reply_body

        return output

    async def digest_all(self):
        query = """
            SELECT
                fe.funding_request_id,
                fe.professor_reply_body,
                fe.main_email_body,
                fe.professor_name,
                fe.main_email_subject
            FROM funding_emails fe
            LEFT JOIN funding_replies fr ON fe.funding_request_id = fr.funding_request_id
            WHERE fe.professor_replied = 1 AND fr.id IS NULL
        """

        rows = await DB.fetch_all(query)

        if not rows:
            print("No new replies to digest.")
            return []

        tasks = [
            self.digest(
                funding_request_id=row["funding_request_id"],
                professor_reply_body=row["professor_reply_body"],
                main_email_body=row["main_email_body"],
                professor_name=row["professor_name"],
                main_email_subject=row["main_email_subject"],
            ) for row in rows
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and count
        successful = [r for r in results if isinstance(r, dict)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        print(f"Digested {len(successful)} new replies. ({len(failed)} failures)")
        return successful

    @staticmethod
    async def upsert_digested(digested: dict):
        query = """
            INSERT INTO funding_replies (
                funding_request_id,
                reply_body_raw,
                reply_body_cleaned,
                needs_human_review,
                is_auto_generated,
                auto_generated_type,
                engagement_label,
                engagement_bool,
                next_step_type,
                activity_status,
                activity_bool,
                short_rationale,
                key_phrases,
                confidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                reply_body_raw = VALUES(reply_body_raw),
                reply_body_cleaned = VALUES(reply_body_cleaned),
                needs_human_review = VALUES(needs_human_review),
                is_auto_generated = VALUES(is_auto_generated),
                auto_generated_type = VALUES(auto_generated_type),
                engagement_label = VALUES(engagement_label),
                engagement_bool = VALUES(engagement_bool),
                next_step_type = VALUES(next_step_type),
                activity_status = VALUES(activity_status),
                activity_bool = VALUES(activity_bool),
                short_rationale = VALUES(short_rationale),
                key_phrases = VALUES(key_phrases),
                confidence = VALUES(confidence)
        """

        # Serialize key_phrases list to JSON string for MySQL JSON column
        key_phrases = digested.get("key_phrases")
        if isinstance(key_phrases, list):
            key_phrases = json.dumps(key_phrases, ensure_ascii=False)
        
        await DB.execute(
            query,
            (
                digested.get("funding_request_id"),
                digested.get("reply_body_raw"),
                digested.get("reply_body_cleaned"),
                digested.get("needs_human_review"),
                digested.get("is_auto_generated"),
                digested.get("auto_generated_type"),
                digested.get("engagement_label"),
                digested.get("engagement_bool"),
                digested.get("next_step_type"),
                digested.get("activity_status"),
                digested.get("activity_bool"),
                digested.get("short_rationale"),
                key_phrases,
                digested.get("confidence"),
            ),
        )

    @staticmethod
    async def upsert_digested_batch(results: list[dict]):
        """Upsert all digested replies in a single transaction (atomic batch)."""
        if not results:
            return

        query = """
            INSERT INTO funding_replies (
                funding_request_id,
                reply_body_raw,
                reply_body_cleaned,
                needs_human_review,
                is_auto_generated,
                auto_generated_type,
                engagement_label,
                engagement_bool,
                next_step_type,
                activity_status,
                activity_bool,
                short_rationale,
                key_phrases,
                confidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                reply_body_raw = VALUES(reply_body_raw),
                reply_body_cleaned = VALUES(reply_body_cleaned),
                needs_human_review = VALUES(needs_human_review),
                is_auto_generated = VALUES(is_auto_generated),
                auto_generated_type = VALUES(auto_generated_type),
                engagement_label = VALUES(engagement_label),
                engagement_bool = VALUES(engagement_bool),
                next_step_type = VALUES(next_step_type),
                activity_status = VALUES(activity_status),
                activity_bool = VALUES(activity_bool),
                short_rationale = VALUES(short_rationale),
                key_phrases = VALUES(key_phrases),
                confidence = VALUES(confidence)
        """

        def _serialize_key_phrases(kp):
            if isinstance(kp, list):
                return json.dumps(kp, ensure_ascii=False)
            return kp

        values = [
            (
                d.get("funding_request_id"),
                d.get("reply_body_raw"),
                d.get("reply_body_cleaned"),
                d.get("needs_human_review"),
                d.get("is_auto_generated"),
                d.get("auto_generated_type"),
                d.get("engagement_label"),
                d.get("engagement_bool"),
                d.get("next_step_type"),
                d.get("activity_status"),
                d.get("activity_bool"),
                d.get("short_rationale"),
                _serialize_key_phrases(d.get("key_phrases")),
                d.get("confidence"),
            )
            for d in results
        ]

        await DB.execute_many_transaction(query, values)
