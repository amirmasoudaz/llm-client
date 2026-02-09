"""
Letter Review Agent - Usage Examples
=====================================

This file demonstrates how to use the letter review agent for evidence-based
evaluation of academic letters with reproducible scoring.
"""

import asyncio
from src.agents.letter.engine import LetterEngine


# Example 1: Basic Letter Review
async def basic_review_example():
    """
    Review a letter and get evidence-based feedback with scores.
    """
    engine = LetterEngine()
    
    # Sample letter (from previous generation)
    letter = {
        "recipient_name": "Dr. Sarah Johnson",
        "recipient_position": "Associate Professor",
        "recipient_institution": "MIT CSAIL",
        "recipient_city": "Cambridge, MA",
        "recipient_country": "USA",
        "signature_name": "Alex Chen",
        "signature_city": "Toronto",
        "signature_country": "Canada",
        "signature_phone": "+1-416-555-0123",
        "signature_email": "alex.chen@example.com",
        "signature_linkedin": "https://linkedin.com/in/alexchen",
        "date": "January 15, 2025",
        "salutation": "Dear Dr. Johnson,",
        "body": """I am writing to express my interest in your PhD program at MIT CSAIL. I have experience in machine learning and computer vision, and I believe your program would be a great fit for my interests.

During my studies, I worked on several projects related to deep learning. I am passionate about research and would like to contribute to your lab.

I am excited about the opportunity to join your research group and look forward to hearing from you.""",
        "closing_valediction": "Sincerely"
    }
    
    # Sender context (for evidence verification)
    sender_detail = {
        "identity": {
            "full_name": "Alex Chen",
            "email": "alex.chen@example.com"
        },
        "education": [
            {
                "degree": "M.Sc. Computer Science",
                "institution": "University of Toronto",
                "year": "2023"
            }
        ],
        "research_experience": [
            {
                "title": "Research Assistant",
                "description": "Worked on CNN-based medical image segmentation",
                "tools": ["PyTorch", "OpenCV"]
            }
        ]
    }
    
    recipient_detail = {
        "name": "Dr. Sarah Johnson",
        "institution": "MIT CSAIL",
        "research_areas": ["Deep Learning", "Medical Image Analysis"]
    }
    
    # Review the letter
    review = await engine.review(
        letter=letter,
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        cache=True
    )
    
    # Display results
    print("=== LETTER REVIEW ===\n")
    
    print("SCORES:")
    for dimension in ['specificity', 'research_fit', 'evidence_quality', 
                      'structure_flow', 'academic_tone', 'technical_depth', 'overall_strength']:
        score_data = review[dimension]
        print(f"  {dimension.replace('_', ' ').title()}: {score_data['score']}/10")
        print(f"    Justification: {score_data['justification']}")
        if score_data.get('evidence'):
            print(f"    Evidence: {score_data['evidence'][0][:100]}...")
        print()
    
    print(f"\nAVERAGE SCORE: {review['average_score']:.2f}/10")
    print(f"READINESS LEVEL: {review['readiness_level']}\n")
    
    print("TOP 3 STRENGTHS:")
    for i, strength in enumerate(review['strengths'][:3], 1):
        print(f"  {i}. {strength}")
    
    print("\nTOP 3 WEAKNESSES:")
    for i, weakness in enumerate(review['weaknesses'][:3], 1):
        print(f"  {i}. {weakness}")
    
    print("\nPRIORITY IMPROVEMENTS:")
    for i, improvement in enumerate(review['priority_improvements'], 1):
        print(f"  {i}. {improvement}")
    
    return review


# Example 2: Test Reproducibility
async def reproducibility_test():
    """
    Verify that the same letter receives the same scores across multiple reviews.
    """
    engine = LetterEngine()
    
    letter = {...}  # Same as above
    sender_detail = {...}
    recipient_detail = {...}
    
    print("=== REPRODUCIBILITY TEST ===\n")
    print("Running review 5 times with identical inputs...\n")
    
    reviews = []
    for i in range(5):
        review = await engine.review(
            letter=letter,
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            cache=False  # Disable cache to test true reproducibility
        )
        reviews.append(review)
        print(f"Run {i+1}: Average Score = {review['average_score']:.2f}")
    
    # Check if all scores are identical
    first_review = reviews[0]
    all_identical = True
    
    for dimension in ['specificity', 'research_fit', 'evidence_quality',
                      'structure_flow', 'academic_tone', 'technical_depth', 'overall_strength']:
        scores = [r[dimension]['score'] for r in reviews]
        if len(set(scores)) > 1:
            all_identical = False
            print(f"\n❌ INCONSISTENCY in {dimension}: {scores}")
        else:
            print(f"✅ {dimension}: {scores[0]}/10 (consistent)")
    
    print(f"\n{'✅ REPRODUCIBILITY VERIFIED' if all_identical else '❌ REPRODUCIBILITY FAILED'}")
    return all_identical


# Example 3: Review → Optimize Workflow
async def review_optimize_workflow():
    """
    Complete workflow: review letter, then optimize based on feedback.
    """
    engine = LetterEngine()
    
    # Initial letter
    letter = {...}  # Your letter
    sender_detail = {...}
    recipient_detail = {...}
    
    # Step 1: Review
    print("=== STEP 1: REVIEW LETTER ===\n")
    review = await engine.review(letter, sender_detail, recipient_detail)
    
    print(f"Initial Score: {review['average_score']:.2f}/10")
    print(f"Readiness: {review['readiness_level']}\n")
    
    # Step 2: Extract optimization suggestions
    if review.get('optimization_suggestions'):
        optimization_feedback = review['optimization_suggestions']
    else:
        # Construct from priority improvements
        optimization_feedback = "\n".join([
            f"{i}. {imp}"
            for i, imp in enumerate(review['priority_improvements'], 1)
        ])
    
    print("=== STEP 2: OPTIMIZE BASED ON REVIEW ===\n")
    
    optimized_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context={
            "old_letter": letter,
            "feedback": optimization_feedback,
            "revision_goals": [
                "address all weaknesses identified",
                "strengthen low-scoring dimensions"
            ]
        }
    )
    
    # Step 3: Review optimized version
    print("=== STEP 3: REVIEW OPTIMIZED LETTER ===\n")
    review_v2 = await engine.review(optimized_letter, sender_detail, recipient_detail)
    
    print(f"Optimized Score: {review_v2['average_score']:.2f}/10")
    print(f"Improvement: +{review_v2['average_score'] - review['average_score']:.2f} points")
    print(f"New Readiness: {review_v2['readiness_level']}\n")
    
    # Compare scores
    print("SCORE COMPARISON:")
    for dimension in ['specificity', 'research_fit', 'evidence_quality',
                      'structure_flow', 'academic_tone', 'technical_depth']:
        old_score = review[dimension]['score']
        new_score = review_v2[dimension]['score']
        delta = new_score - old_score
        emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➖")
        print(f"  {dimension.replace('_', ' ').title()}: {old_score} → {new_score} {emoji}")
    
    return review, review_v2


# Example 4: Batch Letter Review
async def batch_review_example():
    """
    Review multiple letters and compare scores.
    """
    engine = LetterEngine()
    
    letters = [
        {"name": "Letter A (Weak)", "letter": {...}},
        {"name": "Letter B (Medium)", "letter": {...}},
        {"name": "Letter C (Strong)", "letter": {...}},
    ]
    
    sender_detail = {...}
    recipient_detail = {...}
    
    print("=== BATCH LETTER REVIEW ===\n")
    
    results = []
    for item in letters:
        review = await engine.review(
            letter=item["letter"],
            sender_detail=sender_detail,
            recipient_detail=recipient_detail
        )
        results.append({
            "name": item["name"],
            "score": review['average_score'],
            "readiness": review['readiness_level']
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("RANKING:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}: {result['score']:.2f}/10 ({result['readiness']})")
    
    return results


# Example 5: Dimension-Specific Analysis
async def dimension_analysis_example():
    """
    Focus on analyzing specific dimensions in detail.
    """
    engine = LetterEngine()
    
    letter = {...}
    sender_detail = {...}
    recipient_detail = {...}
    
    review = await engine.review(letter, sender_detail, recipient_detail)
    
    print("=== DIMENSION-SPECIFIC ANALYSIS ===\n")
    
    # Identify weakest dimensions
    dimensions = ['specificity', 'research_fit', 'evidence_quality',
                  'structure_flow', 'academic_tone', 'technical_depth']
    
    dimension_scores = [(dim, review[dim]['score']) for dim in dimensions]
    dimension_scores.sort(key=lambda x: x[1])
    
    print("WEAKEST DIMENSIONS (focus here):\n")
    for dim, score in dimension_scores[:3]:
        dim_data = review[dim]
        print(f"{dim.replace('_', ' ').title()}: {score}/10")
        print(f"  Problem: {dim_data['justification']}")
        print(f"  Evidence: {dim_data.get('evidence', ['N/A'])[0][:150]}...")
        print(f"  Suggestions:")
        for suggestion in dim_data.get('suggestions', []):
            print(f"    - {suggestion}")
        print()
    
    return review


# Example 6: Export Review to Optimization Context
async def export_for_optimization():
    """
    Generate structured feedback that can be directly used for optimization.
    """
    engine = LetterEngine()
    
    letter = {...}
    sender_detail = {...}
    recipient_detail = {...}
    
    review = await engine.review(letter, sender_detail, recipient_detail)
    
    # Build structured optimization context
    optimization_context = {
        "old_letter": letter,
        "feedback": {},
        "revision_goals": []
    }
    
    # Extract dimension-specific feedback
    for dimension in ['specificity', 'research_fit', 'evidence_quality',
                      'structure_flow', 'academic_tone', 'technical_depth']:
        dim_data = review[dimension]
        if dim_data['score'] < 7:  # Below "good" threshold
            optimization_context["feedback"][dimension] = {
                "score": dim_data['score'],
                "issues": dim_data['justification'],
                "suggestions": dim_data['suggestions']
            }
            optimization_context["revision_goals"].append(f"improve {dimension}")
    
    # Add priority improvements
    optimization_context["priority_actions"] = review['priority_improvements']
    
    print("=== OPTIMIZATION CONTEXT (Ready for optimization agent) ===\n")
    print(json.dumps(optimization_context, indent=2))
    
    return optimization_context


# Run examples
if __name__ == "__main__":
    import json
    
    # Run basic review
    asyncio.run(basic_review_example())
    
    # Test reproducibility
    # asyncio.run(reproducibility_test())
    
    # Full workflow: review → optimize → review
    # asyncio.run(review_optimize_workflow())
    
    # Batch review
    # asyncio.run(batch_review_example())
    
    # Dimension analysis
    # asyncio.run(dimension_analysis_example())
    
    # Export for optimization
    # asyncio.run(export_for_optimization())
