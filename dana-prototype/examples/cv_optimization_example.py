"""
Example script demonstrating selective CV optimization using the CVEngine.

This script optimizes specific sections of a CV based on feedback while preserving others.
"""

import asyncio
import json
from src.agents.resume.engine import CVEngine


async def main():
    engine = CVEngine()
    
    # Original CV to optimize
    current_cv = {
        "basics": {
            "full_name": "Arousha Ahmadi",
            "degrees": ["DVM"],
            "current_title": "Doctor of Veterinary Medicine (DVM)",
            "current_affiliation": "Shahid Bahonar University of Kerman",
            "location": {"city": "Tehran", "country": "Iran"},
            "email": "arousha.ahmadi@gmail.com",
            "phone": "+98 921 327 2744",
            "linkedin": "https://www.linkedin.com/in/arousha-ahmadi/"
        },
        "sections": {
            "profile": "Veterinary professional with research experience.",  # Too brief
            "education": [
                {
                    "degree": "DVM, Doctor of Veterinary Medicine",
                    "institution": "Shahid Bahonar University of Kerman",
                    "location": {"city": "Kerman", "country": "Iran"},
                    "date_range": {"start": "2018", "end": "2025"},
                    "gpa": {"score": 18.34, "scale": "20"},
                    "ranking": "Ranked 1st in entering cohort (2018)",
                    "thesis_title": "Innovative application of iron-enriched Saccharomyces boulardii in yogurt fortification"
                }
            ],
            "research_positions": [
                {
                    "title": "Thesis Researcher",
                    "organization": "Shahid Bahonar University of Kerman",
                    "department_or_unit": "Department of Food Hygiene",
                    "date_range": {"start": "2023", "end": "2025"},
                    "position_category": "Research",
                    "bullets": [
                        "Did research on fermentation.",  # Too vague
                        "Worked with bacteria."  # Too vague
                    ]
                }
            ],
            "publications": [
                {
                    "title": "Antibacterial activity of dromedary camel milk",
                    "authors": ["Ahmadi, Arousha"],
                    "venue": "J Vet Comp Biomed Res",
                    "year": 2024,
                    "status": "Published",
                    "publication_type": "Journal article"
                }
            ],
            "skills": [
                {"topic": "Lab skills", "skills": ["PCR", "staining"]}
            ],
            "research_interests": [
                "Microbiology"  # Too vague
            ]
        }
    }
    
    # User details for fact-checking (what info can be used)
    user_details = {
        "research_experience": {
            "thesis_work": {
                "description": "21-day controlled fermentation ecosystem study",
                "methods": ["DPPH", "TBARS", "ferrozine iron assay", "CIELAB colorimetry"],
                "organisms": ["S. boulardii", "L. acidophilus", "Bifidobacterium", "S. thermophilus"],
                "outcomes": "Maintained microbial viability at 7.9-8.4 log CFU/mL",
                "statistics": "ANOVA (p<0.05)"
            }
        },
        "skills_detailed": {
            "Microbial ecology": ["population dynamics", "community stability", "environmental modulation"],
            "Laboratory microbiology": ["culture-based enumeration", "anaerobic handling", "PCR", "Gram staining"],
            "Metabolic assays": ["DPPH", "TBARS", "ferrozine iron assay", "titratable acidity", "pH", "colorimetry"],
            "Data analysis": ["ANOVA", "regression", "Python", "SPSS", "Excel"]
        },
        "research_interests_detailed": [
            "Gut microbial ecology and host-microbe interactions",
            "Evolution and adaptation of microbial communities",
            "Diet- and environment-driven modulation of microbial ecosystems",
            "Translational microbiome science"
        ]
    }
    
    # Sections to modify (only these will change)
    sections_to_modify = ["profile", "research_positions", "skills", "research_interests"]
    
    # Feedback for optimization
    feedback = """
    1. Profile is too brief - expand to 2-3 sentences highlighting research focus, methodology expertise, and goals.
    2. Research position bullets are too vague - add specific methodologies (DPPH, TBARS, ferrozine assays), 
       organism names (S. boulardii, L. acidophilus), and quantified outcomes (7.9-8.4 log CFU/mL).
    3. Skills section is incomplete - add all skill categories from user details with specific tools.
    4. Research interests are too vague - expand to specific research themes from user details.
    """
    
    # Revision goals
    revision_goals = [
        "Strengthen specificity with metrics and methodologies",
        "Expand profile to 2-3 compelling sentences",
        "Add comprehensive skill categories"
    ]
    
    print("Optimizing CV sections...")
    print(f"Sections to modify: {sections_to_modify}\n")
    
    result = await engine.optimize(
        cv=current_cv,
        sections_to_modify=sections_to_modify,
        feedback=feedback,
        user_details=user_details,
        revision_goals=revision_goals,
        cache=False
    )
    
    if not result:
        print("Failed to optimize CV!")
        return
    
    # Display results
    print("=" * 80)
    print("OPTIMIZED CV")
    print("=" * 80)
    
    optimized_cv = result.get("cv", {})
    sections = optimized_cv.get("sections", {})
    
    # Show optimized sections
    print("\n-- PROFILE (optimized) --")
    print(sections.get("profile", "N/A"))
    
    print("\n-- RESEARCH POSITIONS (optimized bullets) --")
    for pos in sections.get("research_positions", []):
        print(f"\n{pos.get('title', 'N/A')}:")
        for bullet in pos.get("bullets", []):
            print(f"  • {bullet}")
    
    print("\n-- SKILLS (optimized) --")
    for skill in sections.get("skills", []):
        skills_list = ", ".join(skill.get("skills", []))
        print(f"  {skill.get('topic', 'N/A')}: {skills_list}")
    
    print("\n-- RESEARCH INTERESTS (optimized) --")
    for interest in sections.get("research_interests", []):
        print(f"  • {interest}")
    
    print("\n-- EDUCATION (should be unchanged) --")
    for edu in sections.get("education", []):
        print(f"  {edu.get('degree', 'N/A')} - {edu.get('institution', 'N/A')}")
    
    print("\n" + "=" * 80)
    print(f"Modified sections: {result.get('modified_sections', [])}")


if __name__ == "__main__":
    asyncio.run(main())
