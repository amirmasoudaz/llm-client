"""
Example script demonstrating CV review using the CVEngine.

This script reviews an academic CV and provides multi-dimensional feedback with scores.
"""

import asyncio
import json
from src.agents.resume.engine import CVEngine


async def main():
    engine = CVEngine()
    
    # Sample CV to review (AcademicCV format)
    cv_to_review = {
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
            "profile": "Veterinary professional with research experience in microbial ecology, fermentation systems, and host-microbe interactions. Trained in controlled ecosystem experiments, microbial community monitoring, and biochemical response analysis.",
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
                        "Designed and executed a 21-day controlled fermentation ecosystem.",
                        "Investigated ecological interactions among probiotic species.",
                        "Assessed microbial metabolic responses using DPPH, TBARS, and ferrozine assays."
                    ]
                }
            ],
            "publications": [
                {
                    "title": "Antibacterial activity of dromedary camel milk fermented with probiotics",
                    "authors": ["Ahmadi, Arousha"],
                    "venue": "Journal of Veterinary and Comparative Biomedical Research",
                    "year": 2024,
                    "status": "Published",
                    "publication_type": "Journal article"
                }
            ],
            "skills": [
                {"topic": "Microbial ecology", "skills": ["population dynamics", "community stability"]},
                {"topic": "Laboratory microbiology", "skills": ["culture-based enumeration", "PCR", "staining"]}
            ],
            "research_interests": [
                "Gut microbial ecology and host-microbe interactions",
                "Evolution and adaptation of microbial communities"
            ],
            "languages": [
                {"language": "Persian", "level": "Native"},
                {"language": "English", "level": "Advanced"}
            ]
        }
    }
    
    # Target context for fit evaluation
    target_context = {
        "position": "PhD position in Microbial Ecology",
        "institution": "University of Alberta",
        "lab": "Walter Lab - Gut Microbiome Research",
        "research_areas": ["gut microbiome", "host-microbe interactions", "microbial ecology"]
    }
    
    # Review the CV
    print("Reviewing CV...\n")
    
    review = await engine.review(
        cv=cv_to_review,
        target_context=target_context,
        cache=False
    )
    
    if not review:
        print("Failed to generate review!")
        return
    
    # Display results
    print("=" * 80)
    print("CV REVIEW RESULTS")
    print("=" * 80)
    
    # Dimension scores
    dimensions = review.get("dimensions", {})
    print("\nDimension Scores:")
    for dim_name, dim_data in dimensions.items():
        if isinstance(dim_data, dict):
            score = dim_data.get("score", "N/A")
            print(f"  {dim_name}: {score}/10")
    
    # Summary
    print(f"\nAverage Score: {review.get('average_score', 'N/A')}")
    print(f"Readiness Level: {review.get('readiness_level', 'N/A')}")
    
    # Strengths
    print("\nStrengths:")
    for strength in review.get("strengths", []):
        print(f"  • {strength}")
    
    # Weaknesses
    print("\nWeaknesses:")
    for weakness in review.get("weaknesses", []):
        print(f"  • {weakness}")
    
    # Priority improvements
    print("\nPriority Improvements:")
    for i, improvement in enumerate(review.get("priority_improvements", []), 1):
        print(f"  {i}. {improvement}")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
