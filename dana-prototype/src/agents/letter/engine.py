# src/generators/letter/engine.py

from typing import List, Optional, Literal, Dict, Tuple, Any
import json, re
from pathlib import Path
import asyncio
import shutil
from datetime import datetime

import aiofiles
import blake3
from pypdf import PdfReader
from llm_client import OpenAIClient, GPT5Mini

from agents.letter.letter_generation import LetterGenerationRespSchema
from agents.letter.letter_review import LetterReviewRespSchema
from src.tools.s3_bootstrap import CPD
from src.agents.letter.context import (
    FROM_SCRATCH_PROMPT_SKELETON,
    OPTIMIZATION_PROMPT_SKELETON,
    REVIEW_PROMPT_SKELETON,
    TONE_MODULES, TAILOR_MODULES,
    DEFAULT_STYLE_ADDONS,
    Tone, Tailor
)


class LetterEngine:
    cache_dir = CPD / "cache" / "letter_generation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = CPD / "artifacts" / "letters"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _UNICODE_LATEX_MAP = {
        "×": "\\times",
        "–": "--",
        "—": "---",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
    }

    READINESS_THRESHOLDS = {
        "needs_major_revision": (1.0, 4.99),
        "needs_minor_revision": (5.0, 6.99),
        "strong": (7.0, 8.49),
        "excellent": (8.5, 10.0)
    }

    def __init__(self) -> None:
        self.openai = OpenAIClient(GPT5Mini, cache_backend="pg_redis", cache_collection="letter_generation")
        self.letter: Optional[dict] = None  # Renamed from self.cover_letter for generality

    def _compute_readiness_level(self, dimensions: dict) -> Tuple[str, float]:
        """Compute readiness level from average score."""
        average = sum(dimensions["scores"].values()) / len(dimensions["scores"])
        for level, (min_score, max_score) in self.READINESS_THRESHOLDS.items():
            if min_score <= average <= max_score:
                return level, average
        return "needs_major_revision", average

    @staticmethod
    def _expand_tailor(tailor: Optional[List[Tailor]]) -> List[Tailor]:
        if not tailor:
            return []
        if "match_everything" in tailor:
            return [
                "match_title",
                "match_location",
                "match_skills",
                "match_experience",
                "match_culture",
            ]
        return tailor

    @staticmethod
    def _normalize_avoid_focus(avoid: Optional[Any], focus: Optional[Any]) -> Tuple[str, str]:
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
            "AVOID: Do not mention these topics unless strictly required by the posting; if unavoidable, reframe briefly and positively: "
            + "; ".join(avoid_list)
            if avoid_list
            else ""
        )
        focus_text = (
            "FOCUS: Emphasize these themes with evidence from the resume/profile (no fabrication): "
            + "; ".join(focus_list)
            if focus_list
            else ""
        )
        return avoid_text, focus_text

    def build_messages(
            self,
            sender_detail: Dict[str, Any],   # profile + resume
            recipient_detail: Dict[str, Any],
            tone: Tone = "formal",
            tailor_type: Optional[List[Tailor]] = None,
            avoid: Optional[Any] = None,
            focus: Optional[Any] = None,
            action_type: Literal["from_scratch", "optimization"] = "from_scratch",
            optimization_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
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

        system_prompt = "\n\n".join([m for m in modules if m])

        contexts = {
            "SENDER_DETAIL_JSON": json.dumps(sender_detail, ensure_ascii=False, default=str),
            "RECIPIENT_DETAIL_JSON": json.dumps(recipient_detail, ensure_ascii=False, default=str),
        }
        if action_type == "optimization":
            contexts["OPTIMIZATION_CONTEXT_JSON"] = json.dumps(optimization_context or {}, ensure_ascii=False, default=str)

        user_payload = "\n\n\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in contexts.items()
        ])

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

    def _normalize_unicode_for_latex(self, s: str) -> str:
        for k, v in self._UNICODE_LATEX_MAP.items():
            if k in s:
                s = s.replace(k, v)
        return s

    @staticmethod
    def _escape_unescaped_latex(s: str) -> str:
        s = re.sub(r"(?<!\\)%", r"\%", s)
        s = re.sub(r"(?<!\\)#", r"\#", s)
        s = re.sub(r"(?<!\\)&", r"\&", s)
        s = re.sub(r"(?<!\\)\$", r"\$", s)
        s = re.sub(r"(?<!\\)_", r"\_", s)
        s = re.sub(r"(?<!\\){", r"\{", s)
        s = re.sub(r"(?<!\\)}", r"\}", s)
        s = re.sub(r"(?<!\\)\^", r"\^{}", s)
        s = re.sub(r"(?<!\\)~", r"\~{}", s)
        return s

    def fmt(self, s: str) -> str:
        return self._escape_unescaped_latex(self._normalize_unicode_for_latex(s or ""))

    @staticmethod
    def _strip_name_from_valediction(val: Optional[str], full_name: Optional[str]) -> Optional[str]:
        if not val:
            return val
        cleaned = val.strip()
        if "," in cleaned:
            cleaned = cleaned.split(",", 1)[0].strip()
        if full_name:
            pattern = re.compile(re.escape(full_name), flags=re.IGNORECASE)
            cleaned = pattern.sub("", cleaned).strip()
        return cleaned

    @staticmethod
    def _remove_duplicate_timezone(p1: Optional[str], p4: Optional[str]) -> Optional[str]:
        if not p1 or not p4:
            return p4
        tz_keywords = ["EST", "CST", "ET", "Eastern", "time zone", "timezone", "time-zone"]
        p1_l = p1.lower()
        p4_l = p4.lower()
        if any(k.lower() in p1_l for k in tz_keywords) and any(k.lower() in p4_l for k in tz_keywords):
            sentences = re.split(r"(?<=[.!?])\s+", p4)
            keep = [s for s in sentences if not any(k.lower() in s.lower() for k in tz_keywords)]
            new_p4 = " ".join(keep).strip()
            return new_p4 if new_p4 else None
        return p4

    def _post_validate(self, output: Dict[str, Any], sender_detail: Dict[str, Any]) -> Dict[str, Any]:
        if not output:
            return output
        full_name = sender_detail.get("identity", {}).get("full_name")

        for key, val in list(output.items()):
            if isinstance(val, str):
                output[key] = self._normalize_unicode_for_latex(val)
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, dict):
                        new_item = {k: self._normalize_unicode_for_latex(v) if isinstance(v, str) else v for k, v in
                                    item.items()}
                        new_list.append(new_item)
                    else:
                        new_list.append(item)
                output[key] = new_list

        output["closing_valediction"] = self._strip_name_from_valediction(output.get("closing_valediction"),
                                                                          full_name)

        return output

    @staticmethod
    async def _read_cache(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    @staticmethod
    async def _write_cache(path: Path, content: dict) -> None:
        if content is None:
            return
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(content, ensure_ascii=False, indent=2))

    def _cache_key(self, messages: list) -> str:
        h = blake3.blake3()
        data = json.dumps(messages, sort_keys=True).encode()
        h.update(data)
        meta = json.dumps({"model": self.openai.model.key}, sort_keys=True)
        h.update(meta.encode())
        return h.hexdigest()

    @staticmethod
    def _slug(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")
        return s.lower() or "letter"

    @staticmethod
    def _pick_engine() -> Dict[str, str]:
        if shutil.which("latexmk"):
            return {"tool": "latexmk", "mode": "latexmk"}
        if shutil.which("xelatex"):
            return {"tool": "xelatex", "mode": "xelatex"}
        if shutil.which("pdflatex"):
            return {"tool": "pdflatex", "mode": "pdflatex"}
        raise RuntimeError("No LaTeX engine found: please install latexmk, xelatex, or pdflatex.")

    @staticmethod
    async def _write_file(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(text)

    async def generate(
            self,
            user_id: str,
            sender_detail: dict,
            recipient_detail: dict,
            tone: Literal[
                "formal",
                "informal",
                "friendly",
                "enthusiastic"
            ] = "formal",
            tailor_type: List[Literal[
                "match_title",
                "match_location",
                "match_skills",
                "match_experience",
                "match_culture",
                "match_everything"
            ]] = None,
            avoid: Any = None,
            focus: Any = None,
            cache: bool = True,
            regenerate: bool = False,
            generation_type: Literal["from_scratch", "optimization"] = "from_scratch",
            optimization_context: Optional[Dict[str, Any]] = None,
    ):
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
            letter = await self._read_cache(cache_path)
            if letter is not None:
                return letter
        elif cache_path.exists() and regenerate:
            cache_path = cache_path.with_name(f"{generation_type}.{key}.regen.json")

        response = await self.openai.get_response(
            messages=messages,
            identifier=f"{generation_type}_{user_id}_{key}",
            response_format=LetterGenerationRespSchema,
            cache_response=cache,
        )
        letter = response.get("output", {})
        if not letter:
            print(f"Failed to generate {generation_type} letter.")

        letter = self._post_validate(letter, sender_detail)

        letter["key"] = key
        letter["cache_path"] = str(cache_path)

        if cache:
            await self._write_cache(cache_path, letter)

        return letter

    async def render(
            self,
            letter: dict,
            letter_type: Literal["sop"] = "sop",
            compile_pdf: bool = True,
            include_project_urls: bool = False,
            out_dir: Optional[Path] = None,
            margin: float = 1.0,
            min_margin: float = 0.50,
            step: float = 0.05,
            max_passes: int = 12,
    ) -> dict:
        letter["latex"] = self._build_tex(letter, margin)

        out_dir = out_dir or (self.artifacts_dir / datetime.now().strftime("%Y-%m-%d"))
        
        # Build filename based on letter type
        institution_slug = self._slug(letter.get("recipient_institution", "institution"))
        name_slug = self._slug(letter.get("recipient_name", "recipient"))
        base = f"sop__{institution_slug}__{name_slug}__{letter["key"][:10]}"

        tex_path = (out_dir or self.artifacts_dir)
        tex_path.mkdir(parents=True, exist_ok=True)
        tex_file = tex_path / f"{base}.tex"
        await self._write_file(tex_file, letter["latex"])
        letter.update({
            "compile_status": "pending",
            "tex_path": str(tex_file),
            "pdf_path": None,
            "log_path": None,
            "latex_engine": None,
            "compile_errors": None,
        })

        if compile_pdf:
            passes = 0
            current_margin = margin

            while passes < max_passes and current_margin >= min_margin:
                letter["latex"] = self._build_tex(letter, current_margin)

                compile_info = await self._compile_tex(letter, basename=base, out_dir=out_dir)
                letter.update({
                    "tex_path": compile_info["tex_path"],
                    "pdf_path": compile_info["pdf_path"],
                    "log_path": compile_info["log_path"],
                    "latex_engine": compile_info["engine"],
                    "compile_errors": compile_info["errors"],
                    "compile_status": compile_info["status"],
                })

                if compile_info["status"] != "ok" or not compile_info["pdf_path"]:
                    break

                pdf_path = Path(compile_info["pdf_path"])
                log_path = Path(compile_info["log_path"]) if compile_info["log_path"] else None
                pages = await self._count_pdf_pages(pdf_path, log_path if log_path else pdf_path.with_suffix(".log"))
                letter["page_count"] = pages
                letter["final_margin"] = current_margin

                if pages == 1:
                    break

                current_margin = round(max(min_margin, current_margin - step), 2)
                passes += 1

        await self._write_cache(Path(letter["cache_path"]), letter)

        return letter

    def _build_tex(self, letter: dict, margin: float = 1.0) -> str:
        """Build LaTeX for Statement of Purpose letter."""
        # Format all text fields
        for key in ["recipient_name", "recipient_position", "recipient_institution", "recipient_city", 
                    "recipient_country", "signature_name", "signature_city", "signature_country", 
                    "signature_phone", "signature_email", "date", "salutation", "body", "closing_valediction"]:
            if key in letter and letter[key] is not None:
                letter[key] = self.fmt(letter[key])
        
        # Ensure salutation ends with comma
        if letter.get("salutation") and not letter["salutation"].strip().endswith(","):
            letter["salutation"] = letter["salutation"].strip() + ","
        
        # Build LinkedIn link if present
        linkedin_link = ""
        if letter.get("signature_linkedin"):
            linkedin_link = rf"\href{{{letter['signature_linkedin']}}}{{LinkedIn}}"
        
        # Build the custom opening with side-by-side layout
        latex = rf"""\documentclass[11pt]{{letter}}
\usepackage[top={margin}in, bottom=0.1in, left={margin}in, right={margin}in]{{geometry}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=black, urlcolor=blue}}

\makeatletter
\renewcommand*{{\opening}}[1]{{%
  \noindent
  \begin{{minipage}}[t]{{0.6\textwidth}}
    \raggedright
    {{\bfseries {letter['recipient_name']}}}\\
    {letter['recipient_position']}\\
    {letter['recipient_institution']}\\
    {letter['recipient_city']}, {letter['recipient_country']}
  \end{{minipage}}%
  \hfill
  \begin{{minipage}}[t]{{0.40\textwidth}}
    \raggedleft
    \textbf{{{letter['signature_name']}}}\\
    {letter['signature_city']}, {letter['signature_country']}\\
    {letter['signature_phone']}\\
    \href{{mailto:{letter['signature_email']}}}{{Email}}"""

        if linkedin_link:
            latex += rf""" \;•\;
    {linkedin_link}"""
        
        latex += rf"""\\[0.5em]
    {letter['date']}
  \end{{minipage}}\\[2em]
  #1\par\nobreak
  \vskip 1.5em
}}
\makeatother

\begin{{document}}

\begin{{letter}}{{}}
\opening{{{letter['salutation']}}}
\vspace{{-13pt}}
{letter['body']}
\closing{{{letter['closing_valediction']},\\[1.5ex]{letter['signature_name']}}}

\end{{letter}}
\end{{document}}
"""
        latex = latex.replace(r"\times", "×")
        return latex

    async def _compile_tex(self, letter: dict, basename: str, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        tex_path = out_dir / f"{basename}.tex"
        pdf_path = out_dir / f"{basename}.pdf"
        log_path = out_dir / f"{basename}.log"

        await self._write_file(tex_path, letter["latex"])

        engine = self._pick_engine()
        cwd = out_dir

        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", tex_path.name] \
            if engine["mode"] == "latexmk" else \
            [engine["tool"], "-interaction=nonstopmode", tex_path.name]

        async def run_once(command):
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode("utf-8", "ignore"), stderr.decode("utf-8", "ignore")

        rc, so, se = await run_once(cmd)
        if rc != 0 and engine["mode"] == "latexmk":
            cmd_xe = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-quiet", tex_path.name]
            rc2, so2, se2 = await run_once(cmd_xe)
            so += "\n[retry with -xelatex]\n" + so2
            se += "\n[retry with -xelatex]\n" + se2
            rc = rc2

        if engine["mode"] in {"xelatex", "pdflatex"} and rc == 0:
            rc2, so2, se2 = await run_once(cmd)
            so += "\n" + so2
            se += "\n" + se2

        await self._write_file(log_path, (so + "\n" + se).strip())

        status = "ok" if (rc == 0 and pdf_path.exists()) else "error"
        errors = None
        if status == "error":
            errors = self._extract_latex_error((so + "\n" + se))

        return {
            "status": status,
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            "log_path": str(log_path),
            "engine": engine["mode"],
            "errors": errors,
        }

    @staticmethod
    def _extract_latex_error(log_text: str) -> str:
        lines = log_text.splitlines()
        snippets = []
        for i, line in enumerate(lines):
            if line.startswith("!"):
                context_before = lines[max(0, i - 2): i]
                context_after = lines[i + 1: i + 3]
                snippets.append("\n".join(context_before + [line] + context_after))
        return "\n---\n".join(snippets[:3]) if snippets else log_text[-2000:]

    @staticmethod
    async def _pdfinfo_pages(pdf_path: Path) -> Optional[int]:
        if not shutil.which("pdfinfo") or not pdf_path.exists():
            return None
        proc = await asyncio.create_subprocess_exec(
            "pdfinfo", str(pdf_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        for line in out.decode("utf-8", "ignore").splitlines():
            if line.lower().startswith("pages:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except Exception:
                    return None
        return None

    @staticmethod
    def _pypdf_pages(pdf_path: Path) -> Optional[int]:
        if PdfReader is None or not pdf_path.exists():
            return None
        try:
            return len(PdfReader(str(pdf_path)).pages)
        except Exception:
            return None

    @staticmethod
    async def _parse_log_pages(log_path: Path) -> Optional[int]:
        if not log_path.exists():
            return None
        async with aiofiles.open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = await f.read()
        m = re.search(r"\( *(\d+)\s+page", text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    async def _count_pdf_pages(self, pdf_path: Path, log_path: Path) -> Optional[int]:
        pages = await self._pdfinfo_pages(pdf_path)
        if pages:
            return pages
        pages = self._pypdf_pages(pdf_path)
        if pages:
            return pages
        return await self._parse_log_pages(log_path)

    def _build_review_messages(
            self,
            letter: Dict[str, Any],
            sender_detail: Dict[str, Any],
            recipient_detail: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build messages for letter review task."""
        
        system_prompt = REVIEW_PROMPT_SKELETON
        
        contexts = {
            "LETTER_TO_REVIEW": json.dumps(letter, ensure_ascii=False, default=str),
            "SENDER_CONTEXT": json.dumps(sender_detail, ensure_ascii=False, default=str),
            "RECIPIENT_CONTEXT": json.dumps(recipient_detail, ensure_ascii=False, default=str),
        }

        user_payload = "\n\n\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in contexts.items()
        ])
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]

    async def review(
            self,
            letter: Dict[str, Any],
            sender_detail: Dict[str, Any],
            recipient_detail: Dict[str, Any],
            cache: bool = True
    ) -> Dict[str, Any]:
        """
        Review a letter and provide evidence-based feedback with scoring.
        
        Args:
            letter: The letter to review (LetterSchema format)
            sender_detail: Applicant's profile and resume
            recipient_detail: Target program/lab information
            cache: Whether to cache the review results
            
        Returns:
            LetterReview dict with scores across 7 dimensions and feedback
        """
        # Build review messages
        messages = self._build_review_messages(letter, sender_detail, recipient_detail)
        
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
            response_format=LetterReviewRespSchema,
            temperature=0,  # CRITICAL: ensures deterministic scoring
            cache_response=cache
        )
        
        review = response.get("output", {})
        if not review:
            print("Failed to generate letter review.")
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
