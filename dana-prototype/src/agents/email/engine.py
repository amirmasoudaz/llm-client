# src/agents/email/engine.py

from typing import List, Optional, Literal, Dict, Tuple, Any
import json
from pathlib import Path

import aiofiles
import blake3
from llm_client import OpenAIClient, GPT5Mini

from agents.email.email_generation import EmailGenerationRespSchema
from agents.email.email_review import EmailReviewRespSchema
from src.tools.s3_bootstrap import CPD
from src.agents.email.context import (
    FROM_SCRATCH_PROMPT_SKELETON,
    OPTIMIZATION_PROMPT_SKELETON,
    REVIEW_PROMPT_SKELETON,
    TONE_MODULES, TAILOR_MODULES,
    DEFAULT_STYLE_ADDONS,
    Tone, Tailor
)



class EmailEngine:
    """Engine for generating, reviewing, and optimizing professor outreach emails."""
    
    cache_dir = CPD / "cache" / "email_generation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    READINESS_THRESHOLDS = {
        "needs_major_revision": (1.0, 4.99),
        "needs_minor_revision": (5.0, 6.99),
        "strong": (7.0, 8.49),
        "excellent": (8.5, 10.0)
    }

    def __init__(self) -> None:
        self.openai = OpenAIClient(GPT5Mini, cache_backend="pg_redis", cache_collection="email_generation")
        self.email: Optional[dict] = None

    def _compute_readiness_level(self, dimensions: dict) -> Tuple[str, float]:
        """Compute readiness level from average score across all dimensions."""
        scores = [
            dimensions["subject_quality"]["score"],
            dimensions["research_fit"]["score"],
            dimensions["evidence_quality"]["score"],
            dimensions["tone_appropriateness"]["score"],
            dimensions["length_efficiency"]["score"],
            dimensions["call_to_action"]["score"],
            dimensions["overall_strength"]["score"],
        ]
        average = sum(scores) / len(scores)

        for level, (min_score, max_score) in self.READINESS_THRESHOLDS.items():
            if min_score <= average <= max_score:
                return level, round(average, 2)

        return "needs_major_revision", round(average, 2)

    @staticmethod
    def _expand_tailor(tailor: Optional[List[Tailor]]) -> List[Tailor]:
        """Expand tailor shortcuts like 'match_everything' into specific tailor types."""
        if not tailor:
            return []
        if "match_everything" in tailor:
            return [
                "match_research_area",
                "match_recent_papers",
                "match_lab_culture",
                "match_collaboration_type",
            ]
        return tailor

    @staticmethod
    def _normalize_avoid_focus(avoid: Optional[Any], focus: Optional[Any]) -> Tuple[str, str]:
        """Convert avoid/focus parameters to formatted instruction strings."""
        def to_list(x: Any) -> List[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            if isinstance(x, dict):
                items: List[str] = []
                for v in x.values():
                    if isinstance(v, str):
                        items.append(v)
                    elif isinstance(v, list):
                        items.extend([str(i) for i in v])
                return items
            if isinstance(x, list):
                return [str(i) for i in x]
            return [str(x)]

        avoid_list = to_list(avoid)
        focus_list = to_list(focus)

        avoid_text = (
            "AVOID: Do not mention these topics unless strictly required by context; if unavoidable, reframe briefly and positively: "
            + "; ".join(avoid_list)
            if avoid_list
            else ""
        )
        focus_text = (
            "FOCUS: Emphasize these themes with evidence from the profile (no fabrication): "
            + "; ".join(focus_list)
            if focus_list
            else ""
        )
        return avoid_text, focus_text

    def build_messages(
            self,
            sender_detail: Dict[str, Any],
            recipient_detail: Dict[str, Any],
            tone: Tone = "formal",
            tailor_type: Optional[List[Tailor]] = None,
            avoid: Optional[Any] = None,
            focus: Optional[Any] = None,
            action_type: Literal["from_scratch", "optimization"] = "from_scratch",
            optimization_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Build LLM messages for email generation or optimization."""
        prompt_skeleton = FROM_SCRATCH_PROMPT_SKELETON if action_type == "from_scratch" else OPTIMIZATION_PROMPT_SKELETON
        modules: List[str] = [prompt_skeleton]

        # Tone
        tone_inst = TONE_MODULES.get(tone, TONE_MODULES["formal"])
        modules.append(tone_inst)

        # Tailoring
        for t in self._expand_tailor(tailor_type):
            modules.append(TAILOR_MODULES[t])

        # Avoid/Focus
        avoid_text, focus_text = self._normalize_avoid_focus(avoid, focus)
        if avoid_text:
            modules.append(avoid_text)
        if focus_text:
            modules.append(focus_text)

        # Style add-ons
        modules.append(DEFAULT_STYLE_ADDONS)

        system_prompt = "\\n\\n".join([m for m in modules if m])

        contexts = {
            "SENDER_DETAIL_JSON": json.dumps(sender_detail, ensure_ascii=False, default=str),
            "RECIPIENT_DETAIL_JSON": json.dumps(recipient_detail, ensure_ascii=False, default=str),
        }
        if action_type == "optimization":
            contexts["OPTIMIZATION_CONTEXT_JSON"] = json.dumps(optimization_context or {}, ensure_ascii=False, default=str)

        user_payload = "\\n\\n\\n".join([
            f"<{k}>\\n{v}\\n</{k}>"
            for k, v in contexts.items()
        ])

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

    @staticmethod
    async def _read_cache(path: Path) -> Optional[dict]:
        """Read cached email from file."""
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    @staticmethod
    async def _write_cache(path: Path, content: dict) -> None:
        """Write email to cache file."""
        if content is None:
            return
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(content, ensure_ascii=False, indent=2))

    def _cache_key(self, messages: list) -> str:
        """Generate unique cache key from messages and model."""
        h = blake3.blake3()
        data = json.dumps(messages, sort_keys=True).encode()
        h.update(data)
        meta = json.dumps({"model": self.openai.model.key}, sort_keys=True)
        h.update(meta.encode())
        return h.hexdigest()

    async def generate(
            self,
            user_id: str,
            sender_detail: dict,
            recipient_detail: dict,
            tone: Literal["formal", "friendly", "enthusiastic"] = "formal",
            tailor_type: Optional[List[Literal[
                "match_research_area",
                "match_recent_papers",
                "match_lab_culture",
                "match_collaboration_type",
                "match_everything"
            ]]] = None,
            avoid: Any = None,
            focus: Any = None,
            cache: bool = True,
            regenerate: bool = False,
            generation_type: Literal["from_scratch", "optimization"] = "from_scratch",
            optimization_context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Generate or optimize a professor outreach email.
        
        Args:
            user_id: Unique identifier for the user
            sender_detail: Dict with sender's profile, research background, achievements
            recipient_detail: Dict with professor's name, position, research areas, etc.
            tone: Email tone (formal, friendly, enthusiastic)
            tailor_type: List of tailoring strategies to apply
            avoid: Topics to avoid mentioning
            focus: Topics to emphasize
            cache: Whether to use/write cache
            regenerate: If True, regenerate even if cached version exists
            generation_type: "from_scratch" or "optimization"
            optimization_context: Context for optimization (old_email, feedback, revision_goals)
            
        Returns:
            Dict with email fields (subject, greeting, body, closing, signatures)
        """
        messages = self.build_messages(
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            tone=tone,
            tailor_type=tailor_type,
            avoid=avoid,
            focus=focus,
            action_type=generation_type,
            optimization_context=optimization_context,
        )

        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"{generation_type}.{key}.json"

        if cache and not regenerate:
            email = await self._read_cache(cache_path)
            if email is not None:
                return email
        elif cache_path.exists() and regenerate:
            cache_path = cache_path.with_name(f"{generation_type}.{key}.regen.json")

        response = await self.openai.get_response(
            messages=messages,
            identifier=f"{generation_type}_{user_id}_{key}",
            response_format=EmailGenerationRespSchema,
            cache_response=cache,
        )
        email = response.get("output", {})
        if not email:
            print(f"Failed to generate {generation_type} email.")
            return {}

        email["key"] = key
        email["cache_path"] = str(cache_path)

        if cache:
            await self._write_cache(cache_path, email)

        return email

    @staticmethod
    def _build_review_messages(
            email: Dict[str, Any],
            sender_detail: Dict[str, Any],
            recipient_detail: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build messages for email review task."""
        
        system_prompt = REVIEW_PROMPT_SKELETON
        
        contexts = {
            "LETTER_TO_REVIEW": json.dumps(email, ensure_ascii=False, default=str),
            "SENDER_CONTEXT": json.dumps(sender_detail, ensure_ascii=False, default=str),
            "RECIPIENT_CONTEXT": json.dumps(recipient_detail, ensure_ascii=False, default=str),
        }

        user_payload = "\\n\\n\\n".join([
            f"<{k}>\\n{v}\\n</{k}>"
            for k, v in contexts.items()
        ])
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]

    async def review(
            self,
            email: Dict[str, Any],
            sender_detail: Dict[str, Any],
            recipient_detail: Dict[str, Any],
            cache: bool = True
    ) -> Dict[str, Any]:
        """
        Review a professor outreach email and provide evidence-based feedback with scoring.
        
        Args:
            email: The email to review (EmailSchema format)
            sender_detail: Sender's profile and research background
            recipient_detail: Professor's research areas and institution info
            cache: Whether to cache the review results
            
        Returns:
            EmailReview dict with scores across 7 dimensions and feedback
        """
        # Build review messages
        messages = self._build_review_messages(email, sender_detail, recipient_detail)
        
        # Generate cache key
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"review.{key}.json"
        
        # Check cache
        if cache:
            cached_review = await self._read_cache(cache_path)
            if cached_review is not None:
                return cached_review
        
        # Call LLM with temperature=0 for reproducibility
        response = await self.openai.get_response(
            messages=messages,
            identifier=f"review_{key}",
            response_format=EmailReviewRespSchema,
            temperature=0,  # CRITICAL: ensures deterministic scoring
            cache_response=cache
        )
        
        review = response.get("output", {})
        if not review:
            print("Failed to generate email review.")
            return {}
        
        readiness_level, average_score = self._compute_readiness_level(review["dimensions"])
        review["readiness_level"] = readiness_level
        review["average_score"] = average_score
        
        # Add metadata
        review["review_key"] = key
        review["cache_path"] = str(cache_path)
        
        # Write to cache
        if cache:
            await self._write_cache(cache_path, review)
        
        return review

    @staticmethod
    def render_html(email: Dict[str, Any]) -> str:
        """
        Render email as simple HTML for viewing or sending.
        
        Args:
            email: Email dict with all fields
            
        Returns:
            HTML string with formatted email
        """
        subject = email.get("subject", "")
        greeting = email.get("greeting", "")
        body = email.get("body", "").replace("\\n\\n", "<br><br>").replace("\\n", "<br>")
        closing = email.get("closing", "")
        signature_name = email.get("signature_name", "")
        signature_email = email.get("signature_email", "")
        signature_phone = email.get("signature_phone", "")
        signature_linkedin = email.get("signature_linkedin", "")
        signature_website = email.get("signature_website", "")
        
        # Build signature
        signature_parts = [signature_name]
        if signature_email:
            signature_parts.append(f'<a href="mailto:{signature_email}">{signature_email}</a>')
        if signature_phone:
            signature_parts.append(signature_phone)
        if signature_linkedin:
            signature_parts.append(f'<a href="{signature_linkedin}">LinkedIn</a>')
        if signature_website:
            signature_parts.append(f'<a href="{signature_website}">Website</a>')
        
        signature = "<br>".join(signature_parts)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 700px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .email-container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .subject {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }}
        .greeting {{
            margin-bottom: 10px;
        }}
        .body {{
            margin-bottom: 20px;
            color: #444;
        }}
        .closing {{
            margin-bottom: 10px;
        }}
        .signature {{
            color: #555;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="email-container">
        <div class="subject">Subject: {subject}</div>
        <div class="greeting">{greeting}</div>
        <div class="body">{body}</div>
        <div class="closing">{closing}</div>
        <div class="signature">{signature}</div>
    </div>
</body>
</html>"""
        return html
