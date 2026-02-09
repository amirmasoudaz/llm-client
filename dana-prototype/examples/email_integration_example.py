"""
Comprehensive integration example demonstrating the full email agent workflow:
1. Generate email from scratch
2. Review the generated email
3. Optimize based on review feedback (if needed)
4. Review again to verify improvements
"""

import asyncio
from src.agents.email.engine import EmailEngine


async def main():
    engine = EmailEngine()
    
    # Setup sender and recipient details
    sender_detail = {
        "identity": {
            "full_name": "Alex Chen",
            "email": "alex.chen@gradschool.edu",
            "phone": "+1-617-555-0123",
            "linkedin": "https://linkedin.com/in/alexchen"
        },
        "current_status": "PhD candidate in Computer Science at State University",
        "research_interests": [
            "neural architecture search",
            "efficient deep learning",
            "computer vision"
        ],
        "background": "4 years of research in deep learning and AutoML",
        "achievements": [
            "First-author CVPR 2024 paper on efficient NAS",
            "Developed NASBench-CV achieving 18% faster search",
            "Best Paper Award at AutoML Workshop 2023"
        ],
        "skills": ["PyTorch", "JAX", "Python", "CUDA", "Docker", "Ray"],
        "thesis_topic": "Efficient Neural Architecture Search for Edge Devices"
    }
    
    recipient_detail = {
        "name": "Dr. Emily Zhang",
        "position": "Associate Professor",
        "institution": "Stanford University",
        "department": "Department of Computer Science",
        "research_areas": [
            "AutoML",
            "neural architecture search",
            "efficient deep learning",
            "edge AI"
        ],
        "recent_work": (
            "Recent ICML 2024 paper on hardware-aware NAS and "
            "ongoing projects on efficient vision transformers for mobile devices"
        ),
        "lab_name": "Efficient AI Lab",
        "lab_culture": "Strong emphasis on reproducible research and open-source tools"
    }
    
    print("\\n" + "=" * 80)
    print("EMAIL AGENT INTEGRATION TEST - FULL WORKFLOW")
    print("=" * 80)
    
    # STEP 1: Generate email from scratch
    print("\\n[STEP 1] Generating email from scratch...")
    print("-" * 80)
    
    email = await engine.generate(
        user_id="integration_test",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",
        tailor_type=["match_research_area", "match_recent_papers"],
        cache=False
    )
    
    print("\\nGenerated Email:")
    print(f"Subject: {email.get('subject')}")
    print(f"\\nBody (first 200 chars): {email.get('body', '')[:200]}...")
    
    # STEP 2: Review the generated email
    print("\\n[STEP 2] Reviewing generated email...")
    print("-" * 80)
    
    review_1 = await engine.review(
        email=email,
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        cache=False
    )
    
    print(f"\\nReadiness Level: {review_1.get('readiness_level', 'N/A').upper()}")
    print(f"Average Score: {review_1.get('average_score', 0):.2f}/10")
    
    print("\\nDimensional Scores:")
    dimensions = review_1.get('dimensions', {})
    for dim_name, dim_data in dimensions.items():
        score = dim_data.get('score', 0)
        print(f"  - {dim_name.replace('_', ' ').title()}: {score}/10")
    
    print("\\nTop 3 Weaknesses:")
    for i, weakness in enumerate(review_1.get('weaknesses', [])[:3], 1):
        print(f"  {i}. {weakness}")
    
    # STEP 3: Optimize if needed (score < 8.0)
    avg_score = review_1.get('average_score', 0)
    if avg_score < 8.0:
        print(f"\\n[STEP 3] Score is {avg_score:.2f} < 8.0, optimizing email...")
        print("-" * 80)
        
        # Build optimization context from review feedback
        optimization_context = {
            "old_email": email,
            "feedback": " ".join(review_1.get('priority_improvements', [])),
            "revision_goals": [
                f"Improve {dim}: {data.get('suggestions', [''])[0]}"
                for dim, data in dimensions.items()
                if data.get('score', 10) < 7
            ]
        }
        
        optimized_email = await engine.generate(
            user_id="integration_test",
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            tone="formal",
            tailor_type=["match_research_area", "match_recent_papers"],
            generation_type="optimization",
            optimization_context=optimization_context,
            cache=False
        )
        
        print("\\nOptimized Email:")
        print(f"Subject: {optimized_email.get('subject')}")
        print(f"\\nBody (first 200 chars): {optimized_email.get('body', '')[:200]}...")
        
        # STEP 4: Review optimized email
        print("\\n[STEP 4] Reviewing optimized email...")
        print("-" * 80)
        
        review_2 = await engine.review(
            email=optimized_email,
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            cache=False
        )
        
        print(f"\\nReadiness Level: {review_2.get('readiness_level', 'N/A').upper()}")
        print(f"Average Score: {review_2.get('average_score', 0):.2f}/10")
        
        print("\\nScore Improvements:")
        dims_2 = review_2.get('dimensions', {})
        for dim_name in dimensions.keys():
            old_score = dimensions[dim_name].get('score', 0)
            new_score = dims_2.get(dim_name, {}).get('score', 0)
            delta = new_score - old_score
            symbol = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  - {dim_name.replace('_', ' ').title()}: {old_score} → {new_score} {symbol}")
        
        final_email = optimized_email
        final_review = review_2
    else:
        print(f"\\n[STEP 3] Score is {avg_score:.2f} >= 8.0, no optimization needed!")
        print("-" * 80)
        final_email = email
        final_review = review_1
    
    # FINAL SUMMARY
    print("\\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\\nFinal Readiness: {final_review.get('readiness_level', 'N/A').upper()}")
    print(f"Final Score: {final_review.get('average_score', 0):.2f}/10")
    
    print("\\nTop 3 Strengths:")
    for i, strength in enumerate(final_review.get('strengths', [])[:3], 1):
        print(f"  {i}. {strength}")
    
    print("\\n" + "-" * 80)
    print("FINAL EMAIL")
    print("-" * 80)
    print(f"\\nSubject: {final_email.get('subject')}")
    print(f"\\n{final_email.get('greeting')}")
    print(f"\\n{final_email.get('body')}")
    print(f"\\n{final_email.get('closing')}")
    print(final_email.get('signature_name'))
    print(final_email.get('signature_email'))
    if final_email.get('signature_phone'):
        print(final_email['signature_phone'])
    
    # Save HTML version
    html = engine.render_html(final_email)
    html_path = "/tmp/final_email.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print("\\n" + "=" * 80)
    print(f"HTML version saved to: {html_path}")
    print("=" * 80 + "\\n")


if __name__ == "__main__":
    asyncio.run(main())
