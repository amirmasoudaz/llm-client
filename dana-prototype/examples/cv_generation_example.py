"""
Example script demonstrating CV generation from scratch using the CVEngine.

This script generates an academic CV based on user details and optional target context.
"""

import asyncio
import json
from src.agents.resume.engine import CVEngine


async def main():
    engine = CVEngine()
    
    # User details (applicant information)
    user_details = {
        "identity": {
            "full_name": "Arousha Ahmadi",
            "email": "arousha.ahmadi@gmail.com",
            "phone": "+98 921 327 2744",
            "linkedin": "https://www.linkedin.com/in/arousha-ahmadi/",
            "location": {
                "city": "Tehran",
                "country": "Iran"
            }
        },
        "current_status": "Doctor of Veterinary Medicine (DVM)",
        "education": [
            {
                "degree": "DVM, Doctor of Veterinary Medicine",
                "institution": "Shahid Bahonar University of Kerman",
                "date_range": {"start": "2018", "end": "2025"},
                "gpa": {"score": 18.34, "scale": "20"},
                "ranking": "Ranked 1st in entering cohort (2018)",
                "thesis_title": "Innovative application of iron-enriched Saccharomyces boulardii in yogurt fortification",
                "thesis_summary": "Thesis defended with distinction; included extensive chemical and microbial tests on antioxidant activity and lipid oxidation."
            }
        ],
        "research_experience": [
            {
                "title": "Thesis Researcher — Probiotic Dairy Fortification",
                "organization": "Shahid Bahonar University of Kerman",
                "department": "Department of Food Hygiene",
                "date_range": {"start": "2023", "end": "2025"},
                "bullets": [
                    "Designed and executed a 21-day controlled fermentation ecosystem to study microbial community stability under nutrient and iron-modulated environments.",
                    "Investigated ecological interactions among S. boulardii, L. acidophilus, Bifidobacterium, and S. thermophilus, quantifying population dynamics, competitive resilience, and community shifts.",
                    "Assessed microbial metabolic responses to environmental stressors using DPPH, TBARS, ferrozine iron assay, and colorimetric profiles (CIELAB L*a*b*)."
                ]
            }
        ],
        "professional_experience": [
            {
                "title": "Scientific Liaison (Veterinary Products)",
                "organization": "Pilvarad Co., Tehran",
                "date_range": {"start": "2025", "end": None},
                "bullets": [
                    "Provide scientific and technical support for veterinary biologicals and vaccines.",
                    "Prepare brochures, slide decks, and technical content; deliver product briefings."
                ]
            }
        ],
        "publications": [
            {
                "title": "Antibacterial activity of dromedary camel milk fermented with probiotics against some pathogenic bacteria",
                "authors": ["Ahmadi, Arousha"],
                "venue": "Journal of Veterinary and Comparative Biomedical Research",
                "year": 2024,
                "status": "Published"
            }
        ],
        "skills": [
            {"topic": "Microbial ecology", "skills": ["population dynamics", "community stability", "environmental modulation"]},
            {"topic": "Laboratory microbiology", "skills": ["culture-based enumeration", "anaerobic handling", "PCR", "staining"]},
            {"topic": "Data analysis", "skills": ["ANOVA", "regression", "Python", "SPSS", "Excel"]}
        ],
        "research_interests": [
            "Gut microbial ecology and host-microbe interactions",
            "Evolution and adaptation of microbial communities",
            "Translational microbiome science"
        ],
        "languages": [
            {"language": "Persian", "level": "Native"},
            {"language": "English", "level": "Advanced"}
        ]
    }
    
    # Optional: Target context for tailoring
    additional_context = {
        "target_position": "PhD position in Microbial Ecology",
        "target_institution": "University of Alberta",
        "target_lab": "Walter Lab - Gut Microbiome Research",
        "emphasis": ["research experience", "microbial ecology skills"]
    }
    
    # Generate CV from scratch
    print("Generating academic CV...\n")
    
    result = await engine.generate(
        user_id="test_user_001",
        user_details=user_details,
        additional_context=additional_context,
        tone="academic",
        cache=False  # Set to False for testing
    )
    
    if not result:
        print("Failed to generate CV!")
        return
    
    # Display the generated CV
    print("=" * 80)
    print("GENERATED CV (JSON)")
    print("=" * 80)
    print(json.dumps(result.get("cv", {}), indent=2, ensure_ascii=False)[:3000])
    print("... (truncated)")
    print("=" * 80)
    
    # Render to LaTeX
    print("\nRendering to LaTeX...")
    tex_content, bib_content = engine.render_latex(result)
    
    print(f"\nTeX content length: {len(tex_content)} characters")
    print(f"BibTeX content length: {len(bib_content)} characters")
    
    # Compile to PDF (requires LaTeX installation)
    print("\nCompiling to PDF...")
    compile_result = await engine.compile_pdf(result)
    
    print(f"Compile status: {compile_result['compile_status']}")
    if compile_result['pdf_path']:
        print(f"PDF saved to: {compile_result['pdf_path']}")
    if compile_result['compile_errors']:
        print(f"Errors: {compile_result['compile_errors'][:500]}")


if __name__ == "__main__":
    asyncio.run(main())
