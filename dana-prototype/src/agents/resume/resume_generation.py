# src/agents/schemas/resume_generation.py

from typing import Union, List, Optional, Literal
from pydantic import BaseModel, Field

JsonPrimitive = Union[str, int, float, bool, None]


class CustomField(BaseModel):
    key: str
    value: Union[JsonPrimitive, List[JsonPrimitive], dict[str, JsonPrimitive]]


class Location(BaseModel):
    city: str
    region: Optional[str] = None
    country: str


class DateRange(BaseModel):
    start: Optional[str] = Field(
        description="ISO date (YYYY or YYYY-MM or YYYY-MM-DD). None allowed if unknown."
    )
    end: Optional[str] = Field(
        None,
        description="ISO date. Use None for ongoing positions (present)."
    )


class GPA(BaseModel):
    score: float
    scale: Literal["4", "10", "20", "100"]


class TechnicalSkill(BaseModel):
    topic: str
    skills: List[str]


class AcademicBasics(BaseModel):
    full_name: str
    degrees: Optional[List[str]] = Field(
        default=None,
        description="Short degree labels, e.g. ['DVM', 'PhD candidate in ...']"
    )
    current_title: Optional[str] = Field(
        default=None,
        description="e.g. 'Doctor of Veterinary Medicine (DVM)' or 'PhD Candidate in ...'"
    )
    current_affiliation: Optional[str] = Field(
        default=None,
        description="e.g. 'Faculty of Veterinary Medicine, Shahid Bahonar University of Kerman'"
    )
    location: Location
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str]
    website: Optional[str]
    orcid: Optional[str]
    scholar_profile: Optional[str]
    other_links: Optional[List[CustomField]]
    custom_fields: Optional[List[CustomField]]


class AcademicEducationItem(BaseModel):
    degree: str  # e.g. "DVM, Doctor of Veterinary Medicine"
    field: Optional[str] = Field(
        default=None,
        description="Major/field if separable, e.g. 'Veterinary Medicine', 'Food Hygiene'"
    )
    institution: str
    location: Optional[Location]
    date_range: DateRange
    gpa: Optional[GPA]
    ranking: Optional[str] = Field(
        default=None,
        description="Cohort ranking or honors, e.g. 'Ranked 1st in entering cohort'"
    )
    thesis_title: Optional[str]
    thesis_supervisor: Optional[str]
    thesis_summary: Optional[str]
    custom_fields: Optional[List[CustomField]]


class AcademicPosition(BaseModel):
    title: str  # "Thesis Researcher", "Scientific Liaison", "Veterinary Internship"
    organization: str  # "Shahid Bahonar University of Kerman", "Pilvarad Co."
    department_or_unit: Optional[str] = Field(
        default=None,
        description="Department, lab, or clinic name"
    )
    location: Optional[Location]
    position_category: Literal[
        "Research",
        "Teaching",
        "Clinical",
        "Professional",
        "Internship",
        "Volunteer",
        "Other",
    ]
    supervisor: Optional[str] = Field(
        default=None,
        description="Main supervisor / PI / mentor, if clearly stated."
    )
    date_range: DateRange
    bullets: List[str]
    custom_fields: Optional[List[CustomField]]


class CourseworkItem(BaseModel):
    title: str  # "Food Industries of Animal, Poultry, and Aquatic Origin"
    institution: Optional[str]
    year: Optional[int]
    grade_string: Optional[str] = Field(
        default=None,
        description="Raw grade representation, e.g. '20.00/20'"
    )
    note: Optional[str] = Field(
        default=None,
        description="Relevance tags like 'high mark', 'relevant to food processing', etc."
    )
    custom_fields: Optional[List[CustomField]]


class PublicationItem(BaseModel):
    title: str
    authors: List[str]
    venue: Optional[str] = Field(
        default=None,
        description="Journal / conference / book / repository"
    )
    year: Optional[int]
    status: Literal[
        "Published",
        "In press",
        "Accepted",
        "Submitted",
        "In preparation",
        "Conference",
        "Preprint",
        "Other",
    ]
    publication_type: Literal[
        "Journal article",
        "Conference paper",
        "Book chapter",
        "Case report",
        "Thesis",
        "Other",
    ]
    doi_or_url: Optional[str]
    citation_string: Optional[str] = Field(
        default=None,
        description="Best-effort formatted citation string, in any standard style."
    )
    notes: Optional[str]
    custom_fields: Optional[List[CustomField]]


class TalkOrPresentationItem(BaseModel):
    title: str
    event_name: str  # "National Feline Congress on Internal Medicine and Surgery"
    year: Optional[int]
    location: Optional[Location]
    role: Optional[str] = Field(
        default=None,
        description="'Invited talk', 'Oral presentation', 'Poster', etc."
    )
    notes: Optional[str]
    custom_fields: Optional[List[CustomField]]


class WorkshopOrCertificationItem(BaseModel):
    title: str  # "Emergency and Trauma Medicine", etc.
    provider: Optional[str]  # "Shahid Bahonar University of Kerman"
    year: Optional[int]
    location: Optional[Location]
    notes: Optional[str]
    custom_fields: Optional[List[CustomField]]


class MembershipItem(BaseModel):
    organization: str  # "Iranian Veterinary Association"
    role: Optional[str] = Field(
        default=None,
        description="e.g. 'Member', 'Student member', 'Volunteer'"
    )
    date_range: Optional[DateRange]
    notes: Optional[str]
    custom_fields: Optional[List[CustomField]]


class LanguageProficiency(BaseModel):
    language: str
    level: Optional[Literal["Native", "Fluent", "Advanced", "Intermediate", "Basic"]]


class LanguageTestScore(BaseModel):
    test_name: str  # "IELTS"
    test_date: Optional[str] = Field(
        default=None,
        description="ISO date if known"
    )
    overall: Optional[float]
    section_scores: Optional[dict[str, float]] = Field(
        default=None,
        description="Mapping from section name to score, e.g. {'Reading': 7.5, 'Writing': 6.5}"
    )
    notes: Optional[str]


class ReferenceItem(BaseModel):
    name: str
    email: Optional[str]
    position: Optional[str]
    organization: Optional[str]
    phone: Optional[str]
    notes: Optional[str]


class AcademicSections(BaseModel):
    profile: Optional[str]
    education: List[AcademicEducationItem]
    research_positions: Optional[List[AcademicPosition]]
    teaching_positions: Optional[List[AcademicPosition]]
    professional_positions: Optional[List[AcademicPosition]]
    clinical_positions: Optional[List[AcademicPosition]]
    coursework: Optional[List[CourseworkItem]]
    publications: Optional[List[PublicationItem]]
    talks_and_presentations: Optional[List[TalkOrPresentationItem]]
    workshops_and_certifications: Optional[List[WorkshopOrCertificationItem]]
    skills: Optional[List[TechnicalSkill]]
    memberships: Optional[List[MembershipItem]]
    languages: Optional[List[LanguageProficiency]]
    language_tests: Optional[List[LanguageTestScore]]
    research_interests: Optional[List[str]]
    references: Optional[List[ReferenceItem]]
    custom_sections: Optional[List[CustomField]]


class CVGenerationRespSchema(BaseModel):
    basics: AcademicBasics
    sections: AcademicSections


class CVOptimizationRespSchema(BaseModel):
    cv: CVGenerationRespSchema = Field(..., description="The structured CV content following AcademicCV schema")
    modified_sections: Optional[List[str]] = Field(
        default=None,
        description="For optimization: list of section names that were modified (e.g., ['profile', 'research_positions'])"
    )


