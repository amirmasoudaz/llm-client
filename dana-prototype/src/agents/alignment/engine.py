# src/agents/alignment/engine.py

from __future__ import annotations

import json, hashlib
from typing import List, Dict, Any, Tuple, Iterable
from pydantic import create_model

from blake3 import blake3
from llm_client import OpenAIClient, GPT5Nano

from src.tools.s3_bootstrap import CPD
from src.agents.alignment.criteria import ALIGNMENT_CRITERIA
from src.agents.alignment.context import SYSTEM_PROMPT
from agents.alignment.alignment_review import (
    QuestionScore, CategoryScore, Judgment,
    AlignmentScore, AlignmentReport, Diagnostics,
    QAResult, LLMQuestionOutputs
)


def _iter_questions() -> Iterable[Tuple[Dict, Dict]]:
    """
    Yields (category_dict, question_dict) for each question in the criteria.
    """
    for cats in ALIGNMENT_CRITERIA.get("alignment_analysis", []):
        for q in (cats.get("questions") or []):
            yield cats, q


def build_question_indexes() -> Tuple[frozenset, Dict[str, Dict], str]:
    """
    Returns:
      REQUIRED_QUESTION_IDS (frozenset),
      QMETA_BY_ID: {qid: {category_id, category_name, weight, question, required}},
      CRITERIA_VERSION: short stable hash of the criteria
    """
    required, qmeta, seen = set(), {}, set()

    for cat, q in _iter_questions():
        qid = q.get("id")
        if not qid:
            raise ValueError(f"Question missing 'id' in category {cat.get('category_id')}")
        if qid in seen:
            raise ValueError(f"Duplicate question id detected: {qid}")
        seen.add(qid)

        is_required = bool(q.get("required", False))

        if is_required:
            required.add(qid)

        qmeta[qid] = {
            "category_id": cat.get("category_id", ""),
            "category_name": cat.get("category_name", ""),
            "weight": float(q.get("weight", 1.0)),
            "question": q.get("question", ""),
            "required": is_required,
        }

    blob = json.dumps(ALIGNMENT_CRITERIA, sort_keys=True, ensure_ascii=False).encode("utf-8")
    criteria_version = hashlib.blake2b(blob, digest_size=8).hexdigest()

    return frozenset(required), qmeta, criteria_version


class AlignmentEngine:
    cache_dir = CPD / "cache" / "alignment_engine"
    cache_dir.mkdir(parents=True, exist_ok=True)

    HIGH_CONF_THRESHOLD = 0.75
    REQUIRED_IDS, QMETA, CRITERIA_VERSION = build_question_indexes()

    TAG_THRESHOLDS = [
        ("EXCELLENT FIT", 85),
        ("GOOD FIT", 70),
        ("MODERATE FIT", 55),
        ("WEAK FIT", 40),
        ("POOR FIT", 0)
    ]

    def __init__(self) -> None:
        self.openai = OpenAIClient(GPT5Nano, cache_backend="pg_redis", cache_collection="alignment_review_engine")

    @staticmethod
    def build_rigid_model():
        """
        Build a structured Pydantic model for LLM output based on ALIGNMENT_CRITERIA.
        Returns the model and an index of (category_id, question_id) tuples.
        """
        cat_models = {}
        index: List[Tuple[str, str]] = []

        for cat in ALIGNMENT_CRITERIA.get("alignment_analysis", []):
            cat_id = cat["category_id"]
            q_fields = {}
            for q in cat.get("questions", []):
                qid = q["id"]
                q_fields[qid] = (Judgment, ...)  # required
                index.append((cat_id, qid))
            cat_model = create_model(f"{cat_id}_Model", **q_fields)
            cat_models[cat_id] = (cat_model, ...)

        TopModel = create_model("RigidLLMOutputs", **cat_models)
        return TopModel, index

    @staticmethod
    def get_textual_alignment_criteria() -> str:
        """Convert ALIGNMENT_CRITERIA to human-readable text format."""
        text = "ALIGNMENT CRITERIA:\n\n"
        for cat in ALIGNMENT_CRITERIA.get("alignment_analysis", []):
            text += f"  {cat['category_id']}: {cat['category_name']}\n\n"
            for q in cat.get("questions", []):
                text += f"    - [{q['id']}] → {q['question']}\n"
                text += f"      Instructions: {q.get('instructions', 'N/A')}\n\n"
        return text.strip()

    @staticmethod
    def build_fill_template() -> Dict[str, Any]:
        """Build template structure for LLM output."""
        tpl = {}
        for cat in ALIGNMENT_CRITERIA.get("alignment_analysis", []):
            cid = cat["category_id"]
            tpl.setdefault(cid, {})
            for q in cat.get("questions", []):
                qid = q["id"]
                tpl[cid][qid] = {
                    "answer": None,
                    "intensity": None,
                    "justification": None,
                }
        return tpl

    @staticmethod
    def flatten_rigid_to_qalist(data) -> List[QAResult]:
        """Flatten structured LLM output to list of QAResult objects."""
        out: List[QAResult] = []
        for cat_id, cat_block in data.items():
            for qid, j in (cat_block or {}).items():
                out.append(QAResult(
                    category_id=cat_id,
                    question_id=qid,
                    answer=j["answer"],
                    intensity=j["intensity"],
                    justification=j["justification"],
                ))
        return out

    def _build_user_payload(
        self,
        user_details: Dict[str, Any],
        professor_profile: Dict[str, Any],
    ) -> str:
        """Build the user message payload for LLM evaluation."""
        professor_text = json.dumps(professor_profile, indent=2, ensure_ascii=False)
        user_text = json.dumps(user_details, indent=2, ensure_ascii=False)
        
        return "\n".join([
            "INPUTS BEGIN",
            "<USER_PROFILE>\n" + user_text + "\n</USER_PROFILE>\n\n",
            "<PROFESSOR_PROFILE>\n" + professor_text + "\n</PROFESSOR_PROFILE>\n\n",
            "<ALIGNMENT_CRITERIA>\n" + self.get_textual_alignment_criteria() + "\n</ALIGNMENT_CRITERIA>",
            "INPUTS END\n",
            "TASK: Answer EVERY question in ALIGNMENT_CRITERIA with the required JSON schema. Do NOT compute any scores.",
        ])

    def _cache_key(self, request_id: str, user_details, professor_profile):
        """Generate cache key for this evaluation request."""
        payload = {
            "request_id": request_id,
            "user": user_details,
            "professor": professor_profile,
            "criteria_version": self.CRITERIA_VERSION,
            "model": self.openai.model.key,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        h = blake3(blob).hexdigest()
        return f"alignment_{h}"

    async def llm_eval(
        self,
        request_id: str,
        user_details: Dict[str, Any],
        professor_profile: Dict[str, Any],
    ) -> LLMQuestionOutputs:
        """Query LLM to evaluate alignment questions."""
        TopModel, _index = self.build_rigid_model()

        content = self._build_user_payload(user_details, professor_profile)
        identifier = self._cache_key(request_id, user_details, professor_profile)
        resp = await self.openai.get_response(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            identifier=identifier,
            response_format=TopModel,
            cache_response=True
        )
        rigid = resp.get("output", resp)
        qa_list = self.flatten_rigid_to_qalist(rigid)
        return LLMQuestionOutputs(results=qa_list)

    @staticmethod
    def _signed(intensity: float, yes: bool) -> float:
        """Convert answer to signed intensity."""
        return float(intensity if yes else -intensity)

    @staticmethod
    def _to_0_100(
            weighted_signed: float,
            weight_sum: float,
            *,
            on_zero: str = "neutral",
            zero_value: float = 50.0,
    ) -> float:
        """
        Normalize from [-weight_sum, +weight_sum] → [0, 100].

        Args:
            weighted_signed: sum over (signed * weight)
            weight_sum: sum over weights
            on_zero: "neutral" → 50.0 when weight_sum≈0, "conservative" → 40.0, "value" → zero_value
            zero_value: used only when on_zero="value"
        """
        EPS = 1e-9
        if weight_sum <= EPS:
            if on_zero == "neutral":
                return 50.0
            if on_zero == "conservative":
                return 40.0
            if on_zero == "value":
                return float(max(0.0, min(100.0, zero_value)))
            raise ValueError("on_zero must be 'neutral', 'conservative', or 'value'.")

        ratio = max(-1.0, min(1.0, weighted_signed / weight_sum))
        return round(50.0 * (1.0 + ratio), 1)

    def _score_categories(
        self,
        qanswers: List[QAResult],
    ) -> Tuple[List[CategoryScore], List[QuestionScore]]:
        """Score each category and return category scores and question scores."""
        # Group questions by category
        by_cat: Dict[str, List[QuestionScore]] = {}
        
        for qa in qanswers:
            meta = self.QMETA.get(qa.question_id, {})
            signed = self._signed(qa.intensity, qa.answer == "Yes")
            qscore = QuestionScore(
                category_id=meta.get("category_id", ""),
                question_id=qa.question_id,
                weight=float(meta.get("weight", 1.0)),
                answer=qa.answer,
                intensity=float(qa.intensity),
                signed=signed,
                justification=qa.justification,
            )
            by_cat.setdefault(meta.get("category_id", ""), []).append(qscore)

        # Compute category scores
        categories: List[CategoryScore] = []
        cat_names: Dict[str, str] = {m["category_id"]: m["category_name"] for m in self.QMETA.values()}

        for cat_id, items in by_cat.items():
            ws, wsum = 0.0, 0.0
            for it in items:
                ws += it.signed * it.weight
                wsum += it.weight
            cat_score = self._to_0_100(ws, wsum, on_zero="conservative")
            categories.append(CategoryScore(
                category_id=cat_id,
                name=cat_names.get(cat_id, cat_id),
                score_0_100=cat_score,
                details=items,
            ))

        all_questions = [q for cat in categories for q in cat.details]
        return sorted(categories, key=lambda c: c.category_id), all_questions

    def _choose_label(self, score: float) -> str:
        """Choose alignment label based on score."""
        for name, th in self.TAG_THRESHOLDS:
            if score >= th:
                return name
        return "POOR FIT"

    def _diag_rollup(self, items) -> Dict[str, float]:
        """Returns high-conf count/rate and mean intensity for a list of QuestionScore."""
        n = len(items)
        if n == 0:
            return {"high_conf_count": 0, "high_conf_rate": 0.0, "mean_intensity": 0.0}
        hc = sum(1 for q in items if q.intensity >= self.HIGH_CONF_THRESHOLD)
        return {
            "high_conf_count": hc,
            "high_conf_rate": round(hc / n, 3),
            "mean_intensity": round(sum(q.intensity for q in items) / n, 3),
        }

    async def evaluate(
        self,
        request_id: str,
        user_details: Dict[str, Any],
        professor_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main entry: LLM judgments → deterministic scoring → final report."""
        
        # Get LLM evaluations
        llm_out = await self.llm_eval(request_id, user_details, professor_profile)
        
        # Score categories
        categories, all_questions = self._score_categories(llm_out.results)
        
        # Calculate overall score
        total_signed, total_w = 0.0, 0.0
        for cat in categories:
            for q in cat.details:
                total_signed += q.signed * q.weight
                total_w += q.weight
        
        overall_score = self._to_0_100(total_signed, total_w, on_zero="neutral")
        overall_score = float(max(0.0, min(100.0, round(overall_score, 1))))
        label = self._choose_label(overall_score)

        # Generate reasons
        reasons: List[str] = []
        
        # Find top positive and negative signals
        pos = sorted([q for q in all_questions if q.signed > 0], 
                     key=lambda x: x.signed * x.weight, reverse=True)
        neg = sorted([q for q in all_questions if q.signed < 0], 
                     key=lambda x: x.signed * x.weight)
        
        for q in (pos[:2] + neg[:2]):
            if q:
                kind = "Strength" if q.signed > 0 else "Weakness"
                reasons.append(f"{kind} • [{q.question_id} · {q.category_id}] {q.justification}")

        # Calculate diagnostics
        required_present = [q for q in all_questions if q.question_id in self.REQUIRED_IDS]
        req_total = len(required_present)
        req_yes = sum(1 for q in required_present if (q.answer == "Yes" and q.intensity >= self.HIGH_CONF_THRESHOLD))
        req_rate = round(req_yes / req_total, 3) if req_total else 0.0

        overall_diag = self._diag_rollup(all_questions)

        diagnostics = Diagnostics(
            coverage_high_conf_count=overall_diag["high_conf_count"],
            coverage_high_conf_rate=overall_diag["high_conf_rate"],
            mean_intensity=overall_diag["mean_intensity"],
            required_yes_rate=req_rate,
            required_yes_count=req_yes,
            required_total=req_total,
        )

        report = AlignmentReport(
            request_id=request_id,
            categories=categories,
            overall=AlignmentScore(score_0_100=overall_score, label=label, reasons=reasons[:6]),
            question_results=all_questions,
            diagnostics=diagnostics
        )
        return report.model_dump()

    async def read_from_db(self):
        """Placeholder for database read operations."""
        pass

    async def write_to_db(self):
        """Placeholder for database write operations."""
        pass
