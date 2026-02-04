# src/agents/programs/engine.py
"""Programs Agent - Professor and program recommendations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from llm_client import OpenAIClient, GPT5Mini

from src.services.db import DatabaseService


@dataclass
class ProfessorFilters:
    """Filters for professor search."""
    countries: Optional[List[str]] = None
    research_areas: Optional[List[str]] = None
    institutions: Optional[List[str]] = None
    min_alignment: float = 0.0
    limit: int = 20


@dataclass
class ProfessorRecommendation:
    """A professor recommendation with alignment score."""
    professor_id: int
    name: str
    institution: str
    department: str
    research_areas: List[str]
    alignment_score: float
    alignment_reasons: List[str]
    email: str
    url: Optional[str] = None
    match_highlights: List[str] = field(default_factory=list)


@dataclass
class ProgramRecommendation:
    """A program/institution recommendation."""
    institution_id: int
    institution_name: str
    department: str
    location: str
    professor_count: int
    avg_alignment: float
    top_professors: List[ProfessorRecommendation]


class ProgramsAgent:
    """
    Agent for recommending professors and programs based on user profile.
    
    Provides:
    - Professor search and ranking by alignment
    - Program/institution recommendations
    - Research area matching
    """
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="programs_agent",
        )
    
    async def recommend_professors(
        self,
        student_id: int,
        filters: Optional[ProfessorFilters] = None,
    ) -> List[ProfessorRecommendation]:
        """
        Recommend professors based on student profile and filters.
        
        Ranks professors by alignment score with the student's research interests.
        """
        filters = filters or ProfessorFilters()
        
        # Get student profile
        profile = await self._get_student_profile(student_id)
        if not profile.get("research_interests"):
            return []
        
        # Build query for professors
        where: Dict[str, Any] = {"is_active": True}
        
        if filters.institutions:
            where["institute"] = {
                "institution_name": {"in": filters.institutions}
            }
        
        if filters.countries:
            where["institute"] = {
                **where.get("institute", {}),
                "country": {"in": filters.countries}
            }
        
        # Get professors
        professors = await self.db.client.fundingprofessor.find_many(
            where=where,
            include={"institute": True},
            take=100,  # Get more than needed for filtering
        )
        
        # Score and rank professors
        scored = []
        for prof in professors:
            score, reasons, highlights = await self._compute_alignment(
                profile, prof
            )
            
            if score >= filters.min_alignment:
                import json
                research_areas = json.loads(prof.research_areas) if isinstance(prof.research_areas, str) else (prof.research_areas or [])
                
                scored.append(ProfessorRecommendation(
                    professor_id=prof.id,
                    name=prof.full_name,
                    institution=prof.institute.institution_name if prof.institute else "",
                    department=prof.department,
                    research_areas=research_areas if isinstance(research_areas, list) else [],
                    alignment_score=score,
                    alignment_reasons=reasons,
                    email=prof.email_address,
                    url=prof.url,
                    match_highlights=highlights,
                ))
        
        # Sort by score
        scored.sort(key=lambda x: x.alignment_score, reverse=True)
        
        # Filter by research areas if specified
        if filters.research_areas:
            filtered = []
            for rec in scored:
                if any(
                    any(area.lower() in ra.lower() for ra in rec.research_areas)
                    for area in filters.research_areas
                ):
                    filtered.append(rec)
            scored = filtered
        
        return scored[:filters.limit]
    
    async def recommend_programs(
        self,
        student_id: int,
        countries: Optional[List[str]] = None,
        min_professors: int = 2,
        limit: int = 10,
    ) -> List[ProgramRecommendation]:
        """
        Recommend programs/institutions based on professor alignment.
        
        Aggregates professor recommendations by institution.
        """
        # Get professor recommendations
        prof_recs = await self.recommend_professors(
            student_id=student_id,
            filters=ProfessorFilters(
                countries=countries,
                limit=100,
            ),
        )
        
        # Group by institution
        by_institution: Dict[int, List[ProfessorRecommendation]] = {}
        institution_info: Dict[int, Dict[str, Any]] = {}
        
        for rec in prof_recs:
            # Get institution ID
            prof = await self.db.client.fundingprofessor.find_unique(
                where={"id": rec.professor_id},
                include={"institute": True},
            )
            if not prof or not prof.institute:
                continue
            
            inst_id = prof.funding_institute_id
            if inst_id not in by_institution:
                by_institution[inst_id] = []
                institution_info[inst_id] = {
                    "name": prof.institute.institution_name,
                    "department": prof.institute.department_name,
                    "city": prof.institute.city,
                    "country": prof.institute.country,
                }
            
            by_institution[inst_id].append(rec)
        
        # Build program recommendations
        programs = []
        for inst_id, profs in by_institution.items():
            if len(profs) < min_professors:
                continue
            
            info = institution_info[inst_id]
            avg_alignment = sum(p.alignment_score for p in profs) / len(profs)
            
            # Sort professors by score
            profs.sort(key=lambda x: x.alignment_score, reverse=True)
            
            location_parts = [info.get("city"), info.get("country")]
            location = ", ".join([p for p in location_parts if p])
            
            programs.append(ProgramRecommendation(
                institution_id=inst_id,
                institution_name=info["name"],
                department=info["department"],
                location=location,
                professor_count=len(profs),
                avg_alignment=avg_alignment,
                top_professors=profs[:5],  # Top 5 professors
            ))
        
        # Sort by average alignment
        programs.sort(key=lambda x: x.avg_alignment, reverse=True)
        
        return programs[:limit]
    
    async def _get_student_profile(self, student_id: int) -> Dict[str, Any]:
        """Get student profile for matching."""
        student = await self.db.client.student.find_unique(
            where={"id": student_id}
        )
        
        if not student:
            return {}
        
        # Get processed resume if exists
        resume = await self.db.client.studentdocument.find_first(
            where={
                "student_id": student_id,
                "document_type": "resume",
                "upload_status": "processed",
            },
            order={"created_at": "desc"},
        )
        
        resume_data = {}
        if resume and resume.processed_content:
            import json
            resume_data = json.loads(resume.processed_content) if isinstance(resume.processed_content, str) else resume.processed_content
        
        return {
            "name": f"{student.first_name} {student.last_name}",
            "research_interests": resume_data.get("research_interests", []),
            "skills": resume_data.get("skills", []),
            "education": resume_data.get("education", []),
            "publications": resume_data.get("publications", []),
        }
    
    async def _compute_alignment(
        self,
        profile: Dict[str, Any],
        professor: Any,
    ) -> tuple[float, List[str], List[str]]:
        """
        Compute alignment score between student and professor.
        
        Returns (score, reasons, highlights).
        """
        import json
        
        # Parse professor data
        research_areas = json.loads(professor.research_areas) if isinstance(professor.research_areas, str) else (professor.research_areas or [])
        if not isinstance(research_areas, list):
            research_areas = []
        
        expertise = json.loads(professor.area_of_expertise) if isinstance(professor.area_of_expertise, str) else (professor.area_of_expertise or [])
        if not isinstance(expertise, list):
            expertise = []
        
        # Student interests
        student_interests = profile.get("research_interests", [])
        student_skills = profile.get("skills", [])
        
        # Simple keyword matching
        score = 0.0
        reasons = []
        highlights = []
        
        # Match research areas
        prof_areas_lower = [a.lower() for a in research_areas]
        for interest in student_interests:
            interest_lower = interest.lower()
            for area in prof_areas_lower:
                if interest_lower in area or area in interest_lower:
                    score += 2.0
                    reasons.append(f"Research area match: {interest}")
                    highlights.append(interest)
                    break
        
        # Match expertise
        expertise_lower = [e.lower() for e in expertise]
        for skill in student_skills:
            skill_lower = skill.lower()
            for exp in expertise_lower:
                if skill_lower in exp or exp in skill_lower:
                    score += 1.0
                    reasons.append(f"Skill/expertise match: {skill}")
                    break
        
        # Normalize score to 0-10 range
        max_possible = len(student_interests) * 2 + len(student_skills)
        if max_possible > 0:
            score = min(10.0, (score / max_possible) * 10)
        else:
            score = 0.0
        
        return score, reasons, list(set(highlights))
    
    async def search_professors(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search professors by name, institution, or research area.
        """
        # Simple text search (in production, would use full-text search)
        query_lower = query.lower()
        
        professors = await self.db.client.fundingprofessor.find_many(
            where={"is_active": True},
            include={"institute": True},
            take=500,
        )
        
        results = []
        for prof in professors:
            import json
            research_areas = json.loads(prof.research_areas) if isinstance(prof.research_areas, str) else (prof.research_areas or [])
            if not isinstance(research_areas, list):
                research_areas = []
            
            # Check for matches
            name_match = query_lower in prof.full_name.lower()
            dept_match = query_lower in prof.department.lower()
            inst_match = prof.institute and query_lower in prof.institute.institution_name.lower()
            area_match = any(query_lower in area.lower() for area in research_areas)
            
            if name_match or dept_match or inst_match or area_match:
                results.append({
                    "professor_id": prof.id,
                    "name": prof.full_name,
                    "department": prof.department,
                    "institution": prof.institute.institution_name if prof.institute else "",
                    "research_areas": research_areas,
                    "email": prof.email_address,
                    "match_type": "name" if name_match else "department" if dept_match else "institution" if inst_match else "research",
                })
        
        return results[:limit]
    
    async def get_professor_details(
        self,
        professor_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a professor."""
        prof = await self.db.client.fundingprofessor.find_unique(
            where={"id": professor_id},
            include={"institute": True},
        )
        
        if not prof:
            return None
        
        import json
        
        return {
            "professor_id": prof.id,
            "name": prof.full_name,
            "title": prof.occupation,
            "department": prof.department,
            "email": prof.email_address,
            "url": prof.url,
            "institution": {
                "name": prof.institute.institution_name if prof.institute else "",
                "department": prof.institute.department_name if prof.institute else "",
                "city": prof.institute.city if prof.institute else "",
                "country": prof.institute.country if prof.institute else "",
            },
            "research_areas": json.loads(prof.research_areas) if isinstance(prof.research_areas, str) else prof.research_areas,
            "expertise": json.loads(prof.area_of_expertise) if isinstance(prof.area_of_expertise, str) else prof.area_of_expertise,
            "credentials": prof.credentials,
        }





