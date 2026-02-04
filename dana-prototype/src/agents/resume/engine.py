# src/agents/resume/engine.py

"""
CV Engine - Main engine for generating, reviewing, and optimizing academic CVs.
"""

from typing import List, Optional, Literal, Dict, Any, Tuple
import json
import re
from pathlib import Path
import asyncio
import shutil
from datetime import datetime

import aiofiles
import blake3
from llm_client import OpenAIClient, GPT5Mini

from agents.resume.resume_generation import CVGenerationRespSchema, CVOptimizationRespSchema
from agents.resume.resume_review import CVReviewRespSchema
from agents.resume.template import CVLatexRenderer
from src.tools.s3_bootstrap import CPD
from src.agents.resume.context import (
    FROM_SCRATCH_PROMPT_SKELETON,
    OPTIMIZATION_PROMPT_SKELETON,
    REVIEW_PROMPT_SKELETON,
    TONE_MODULES, TAILOR_MODULES,
    DEFAULT_STYLE_ADDONS,
    Tone, Tailor
)


class CVEngine:
    """Engine for generating, reviewing, and optimizing academic CVs with LaTeX rendering."""
    
    cache_dir = CPD / "cache" / "cv_generation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts_dir = CPD / "artifacts" / "cvs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    READINESS_THRESHOLDS = {
        "needs_major_revision": (1.0, 4.99),
        "needs_minor_revision": (5.0, 6.99),
        "strong": (7.0, 8.49),
        "excellent": (8.5, 10.0)
    }

    def __init__(self) -> None:
        self.openai = OpenAIClient(GPT5Mini, cache_backend="pg_redis", cache_collection="cv_generation")
        self.cv: Optional[dict] = None
    
    def _compute_readiness_level(self, dimensions: dict) -> Tuple[str, float]:
        """Compute readiness level from average score across all dimensions."""
        scores = [
            dimensions["content_completeness"]["score"],
            dimensions["research_presentation"]["score"],
            dimensions["technical_depth"]["score"],
            dimensions["publication_quality"]["score"],
            dimensions["structure_clarity"]["score"],
            dimensions["target_alignment"]["score"],
            dimensions["overall_strength"]["score"],
        ]
        average = sum(scores) / len(scores)

        for level, (min_score, max_score) in self.READINESS_THRESHOLDS.items():
            if min_score <= average <= max_score:
                return level, round(average, 2)
        return "needs_major_revision", round(average, 2)

    @staticmethod
    def _expand_tailor(tailor: Optional[List[Tailor]]) -> List[Tailor]:
        """Expand tailor shortcuts."""
        if not tailor:
            return []
        return tailor
    
    @staticmethod
    async def _read_cache(path: Path) -> Optional[dict]:
        """Read cached content from file."""
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())
    
    @staticmethod
    async def _write_cache(path: Path, content: dict) -> None:
        """Write content to cache file."""
        if content is None:
            return
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(content, ensure_ascii=False, indent=2, default=str))
    
    def _cache_key(self, messages: list) -> str:
        """Generate unique cache key from messages and model."""
        h = blake3.blake3()
        data = json.dumps(messages, sort_keys=True).encode()
        h.update(data)
        meta = json.dumps({"model": self.openai.model.key}, sort_keys=True)
        h.update(meta.encode())
        return h.hexdigest()
    
    @staticmethod
    def _slug(s: str) -> str:
        """Create URL-safe slug from string."""
        s = re.sub(r"[^A-Za-z0-9]+", "-", s or "").strip("-")
        return s.lower() or "cv"
    
    @staticmethod
    def _pick_engine() -> Dict[str, str]:
        """Select available LaTeX engine."""
        if shutil.which("latexmk"):
            return {"tool": "latexmk", "mode": "latexmk"}
        if shutil.which("xelatex"):
            return {"tool": "xelatex", "mode": "xelatex"}
        if shutil.which("pdflatex"):
            return {"tool": "pdflatex", "mode": "pdflatex"}
        raise RuntimeError("No LaTeX engine found: please install latexmk, xelatex, or pdflatex.")
    
    @staticmethod
    async def _write_file(path: Path, text: str) -> None:
        """Write text to file, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(text)
    
    def _build_generate_messages(
        self,
        user_details: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None,
        tone: Tone = "academic",
        tailor_type: Optional[List[Tailor]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for CV generation."""
        modules: List[str] = [FROM_SCRATCH_PROMPT_SKELETON]
        
        # Tone
        tone_inst = TONE_MODULES.get(tone, TONE_MODULES["academic"])
        modules.append(tone_inst)
        
        # Tailoring
        for t in self._expand_tailor(tailor_type):
            if t in TAILOR_MODULES:
                modules.append(TAILOR_MODULES[t])
        
        # Style add-ons
        modules.append(DEFAULT_STYLE_ADDONS)
        
        system_prompt = "\n\n".join([m for m in modules if m])
        
        contexts = {
            "USER_DETAILS": json.dumps(user_details, ensure_ascii=False, default=str),
        }
        if additional_context:
            contexts["ADDITIONAL_CONTEXT"] = json.dumps(additional_context, ensure_ascii=False, default=str)
        
        user_payload = "\n\n\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in contexts.items()
        ])
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
    
    def _build_optimize_messages(
        self,
        current_cv: Dict[str, Any],
        sections_to_modify: List[str],
        feedback: str,
        user_details: Dict[str, Any],
        revision_goals: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for CV optimization."""
        modules: List[str] = [OPTIMIZATION_PROMPT_SKELETON]
        modules.append(DEFAULT_STYLE_ADDONS)
        
        system_prompt = "\n\n".join([m for m in modules if m])
        
        contexts = {
            "CURRENT_CV": json.dumps(current_cv, ensure_ascii=False, default=str),
            "SECTIONS_TO_MODIFY": json.dumps(sections_to_modify),
            "FEEDBACK": feedback,
            "USER_DETAILS": json.dumps(user_details, ensure_ascii=False, default=str),
        }
        if revision_goals:
            contexts["REVISION_GOALS"] = json.dumps(revision_goals)
        
        user_payload = "\n\n\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in contexts.items()
        ])
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

    @staticmethod
    def _build_review_messages(
        cv: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for CV review."""
        system_prompt = REVIEW_PROMPT_SKELETON
        
        contexts = {
            "CV_TO_REVIEW": json.dumps(cv, ensure_ascii=False, default=str),
        }
        if target_context:
            contexts["TARGET_CONTEXT"] = json.dumps(target_context, ensure_ascii=False, default=str)
        
        user_payload = "\n\n\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in contexts.items()
        ])
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
    
    async def generate(
        self,
        user_id: str,
        user_details: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None,
        tone: Literal["academic", "industry", "clinical"] = "academic",
        tailor_type: Optional[List[Tailor]] = None,
        cache: bool = True,
        regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a complete CV from user details.
        
        Args:
            user_id: Unique identifier for the user
            user_details: Profile information, education, experience, skills, etc.
            additional_context: Target position info, focus areas, emphasis
            tone: CV tone (academic, industry, clinical)
            tailor_type: List of tailoring strategies to apply
            cache: Whether to use/write cache
            regenerate: If True, regenerate even if cached version exists
            
        Returns:
            Dict containing the generated CV in AcademicCV format
        """
        messages = self._build_generate_messages(
            user_details=user_details,
            additional_context=additional_context,
            tone=tone,
            tailor_type=tailor_type,
        )
        
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"generate.{key}.json"
        
        if cache and not regenerate:
            cached = await self._read_cache(cache_path)
            if cached is not None:
                return cached
        elif cache_path.exists() and regenerate:
            cache_path = cache_path.with_name(f"generate.{key}.regen.json")
        
        response = await self.openai.get_response(
            messages=messages,
            identifier=f"cv_generate_{user_id}_{key}",
            response_format=CVGenerationRespSchema,
            cache_response=cache,
        )
        
        cv = response.get("output", {})
        if not cv:
            print("Failed to generate CV.")
            return {}
        
        # Wrap response with metadata
        result = {
            "cv": cv,
            "key": key,
            "cache_path": str(cache_path),
        }
        
        if cache:
            await self._write_cache(cache_path, result)
        
        self.cv = result
        return result
    
    async def review(
        self,
        cv: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None,
        cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Review a CV and provide multi-dimensional feedback with scoring.
        
        Args:
            cv: The CV to review (AcademicCV format, may include wrapper with 'cv' key)
            target_context: Target position/program information for fit evaluation
            cache: Whether to cache the review results
            
        Returns:
            CVReview dict with scores across 7 dimensions and feedback
        """
        # Handle wrapped CV format
        cv_data = cv.get("cv", cv) if isinstance(cv.get("cv"), dict) else cv
        
        messages = self._build_review_messages(cv_data, target_context)
        
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"review.{key}.json"
        
        if cache:
            cached = await self._read_cache(cache_path)
            if cached is not None:
                return cached
        
        response = await self.openai.get_response(
            messages=messages,
            identifier=f"cv_review_{key}",
            response_format=CVReviewRespSchema,
            temperature=0,  # Deterministic scoring
            cache_response=cache,
        )
        
        review = response.get("output", {})
        if not review:
            print("Failed to generate CV review.")
            return {}
        
        # Compute readiness level
        if "dimensions" in review:
            readiness_level, average_score = self._compute_readiness_level(review["dimensions"])
            review["readiness_level"] = readiness_level
            review["average_score"] = average_score
        
        review["review_key"] = key
        review["cache_path"] = str(cache_path)
        
        if cache:
            await self._write_cache(cache_path, review)
        
        return review
    
    async def optimize(
        self,
        cv: Dict[str, Any],
        sections_to_modify: List[str],
        feedback: str,
        user_details: Dict[str, Any],
        revision_goals: Optional[List[str]] = None,
        cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Selectively optimize specified sections of a CV.
        
        Args:
            cv: Current CV (AcademicCV format, may include wrapper with 'cv' key)
            sections_to_modify: List of section names to optimize
            feedback: Specific improvement instructions
            user_details: Original user profile for fact-checking
            revision_goals: Optional list of objectives for this revision
            cache: Whether to cache the result
            
        Returns:
            Dict containing the optimized CV with modified_sections list
        """
        # Handle wrapped CV format
        cv_data = cv.get("cv", cv) if isinstance(cv.get("cv"), dict) else cv
        
        messages = self._build_optimize_messages(
            current_cv=cv_data,
            sections_to_modify=sections_to_modify,
            feedback=feedback,
            user_details=user_details,
            revision_goals=revision_goals,
        )
        
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"optimize.{key}.json"
        
        if cache:
            cached = await self._read_cache(cache_path)
            if cached is not None:
                return cached
        
        response = await self.openai.get_response(
            messages=messages,
            identifier=f"cv_optimize_{key}",
            response_format=CVOptimizationRespSchema,
            cache_response=cache,
        )
        
        optimized_cv = response.get("output", {})
        if not optimized_cv:
            print("Failed to optimize CV.")
            return {}
        
        result = {
            "cv": optimized_cv["cv"],
            "requested_modifications": sections_to_modify,
            "modified_sections": optimized_cv.get("modified_sections", []),
            "key": key,
            "cache_path": str(cache_path),
        }
        
        if cache:
            await self._write_cache(cache_path, result)
        
        self.cv = result
        return result

    @staticmethod
    def render_latex(cv: Dict[str, Any]) -> tuple[str, str, str]:
        """
        Render CV to LaTeX and BibTeX files.
        
        Args:
            cv: CV data (AcademicCV format, may include wrapper with 'cv' key)
            
        Returns:
            Tuple of (tex_content, bib_content)
        """
        cv_data = cv.get("cv", cv) if isinstance(cv.get("cv"), dict) else cv
        renderer = CVLatexRenderer(cv_data)
        return renderer.render()
    
    async def compile_pdf(
        self,
        cv: Dict[str, Any],
        out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Render CV to LaTeX and compile to PDF.
        
        Args:
            cv: CV data (AcademicCV format)
            out_dir: Optional output directory
            
        Returns:
            Dict with tex_path, bib_path, pdf_path, and compilation status
        """
        cv_data = cv.get("cv", cv) if isinstance(cv.get("cv"), dict) else cv
        
        # Generate output paths
        out_dir = out_dir or (self.artifacts_dir / datetime.now().strftime("%Y-%m-%d"))
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from CV owner name
        full_name = cv_data.get("basics", {}).get("full_name", "cv")
        base = self._slug(full_name)
        key = cv.get("key", blake3.blake3(json.dumps(cv_data).encode()).hexdigest()[:10])
        basename = f"{base}_{key[:8]}"
        
        tex_path = out_dir / f"{basename}.tex"
        bib_path = out_dir / "pubs.bib"
        cls_path = out_dir / "resume.cls"
        pdf_path = out_dir / f"{basename}.pdf"
        log_path = out_dir / f"{basename}.log"
        
        # Render LaTeX
        tex_content, bib_content, cls_content = self.render_latex(cv)
        
        await self._write_file(tex_path, tex_content)
        if bib_content:
            await self._write_file(bib_path, bib_content)

        if cls_content:
            await self._write_file(cls_path, cls_content)
        
        result = {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path) if bib_content else None,
            "cls_path": str(cls_path) if cls_content else None,
            "pdf_path": None,
            "log_path": None,
            "compile_status": "pending",
            "compile_errors": None,
        }
        
        # Compile PDF
        try:
            engine = self._pick_engine()
        except RuntimeError as e:
            result["compile_status"] = "error"
            result["compile_errors"] = str(e)
            return result
        
        # Run compilation
        async def run_once(command: list, cwd: Path):
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode("utf-8", "ignore"), stderr.decode("utf-8", "ignore")
        
        cwd = out_dir
        
        if engine["mode"] == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", tex_path.name]
        else:
            cmd = [engine["tool"], "-interaction=nonstopmode", tex_path.name]
        
        rc, so, se = await run_once(cmd, cwd)
        
        # If latexmk failed, try with xelatex
        if rc != 0 and engine["mode"] == "latexmk":
            cmd_xe = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-quiet", tex_path.name]
            rc2, so2, se2 = await run_once(cmd_xe, cwd)
            so += "\n[retry with -xelatex]\n" + so2
            se += "\n[retry with -xelatex]\n" + se2
            rc = rc2
        
        # Second pass for non-latexmk engines
        if engine["mode"] in {"xelatex", "pdflatex"} and rc == 0:
            rc2, so2, se2 = await run_once(cmd, cwd)
            so += "\n" + so2
            se += "\n" + se2
        
        await self._write_file(log_path, (so + "\n" + se).strip())
        
        result["log_path"] = str(log_path)
        
        if rc == 0 and pdf_path.exists():
            result["compile_status"] = "ok"
            result["pdf_path"] = str(pdf_path)
        else:
            result["compile_status"] = "error"
            result["compile_errors"] = self._extract_latex_error(so + "\n" + se)
        
        return result
    
    @staticmethod
    def _extract_latex_error(log_text: str) -> str:
        """Extract error messages from LaTeX log."""
        lines = log_text.splitlines()
        snippets = []
        for i, line in enumerate(lines):
            if line.startswith("!"):
                context_before = lines[max(0, i - 2): i]
                context_after = lines[i + 1: i + 3]
                snippets.append("\n".join(context_before + [line] + context_after))
        return "\n---\n".join(snippets[:3]) if snippets else log_text[-2000:]
