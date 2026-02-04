"""
Example script demonstrating email review functionality using the EmailEngine.

This script reviews a professor outreach email and provides multi-dimensional feedback.
"""

import asyncio
from src.agents.email.engine import EmailEngine


async def main():
    engine = EmailEngine()
    
    # Example email to review (a weak email with common issues)
    email_to_review = {
        "subject": "Research Collaboration Opportunity",
        "greeting": "Dear Dr. Smith,",
        "body": "I am a PhD student interested in machine learning and I would like to work with you. "
                "I have experience in deep learning and have done some projects. "
                "I think your research is very interesting and I would be a good fit for your lab. "
                "Please let me know if you are interested.",
        "closing": "Thanks,",
        "signature_name": "Jane Doe",
        "signature_email": "jane.doe@university.edu"
    }
    
    # Sender context (to verify claims)
    sender_detail = {
        "identity": {"full_name": "Jane Doe"},
        "research_interests": ["machine learning", "deep learning"],
        "background": "PhD student with some ML project experience",
        "achievements": []
    }
    
    # Recipient context (to evaluate fit)
    recipient_detail = {
        "name": "Dr. John Smith",
        "position": "Professor",
        "institution": "MIT CSAIL",
        "research_areas": ["deep learning", "neural architecture search"]
    }
    
    print("Reviewing professor outreach email...\\n")
    
    # Review the email
    review = await engine.review(
        email=email_to_review,
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        cache=False  # Set to False for testing
    )
    
    # Display review results
    print("=" * 80)
    print("EMAIL REVIEW RESULTS")
    print("=" * 80)
    
    print(f"\\nReadiness Level: {review.get('readiness_level', 'N/A').upper()}")
    print(f"Average Score: {review.get('average_score', 'N/A')}/10")
    
    print("\\n" + "-" * 80)
    print("DIMENSIONAL SCORES")
    print("-" * 80)
    
    dimensions = review.get('dimensions', {})
    for dim_name, dim_data in dimensions.items():
        print(f"\\n{dim_name.replace('_', ' ').title()}: {dim_data.get('score', 'N/A')}/10")
        print(f"  Justification: {dim_data.get('justification', 'N/A')}")
        
        evidence = dim_data.get('evidence', [])
        if evidence:
            print(f"  Evidence:")
            for ev in evidence:
                print(f"    - \\\"{ev}\\\"")
        
        suggestions = dim_data.get('suggestions', [])
        if suggestions:
            print(f"  Suggestions:")
            for sug in suggestions:
                print(f"    - {sug}")
    
    print("\\n" + "-" * 80)
    print("SUMMARY FEEDBACK")
    print("-" * 80)
    
    strengths = review.get('strengths', [])
    print("\\nStrengths:")
    for i, strength in enumerate(strengths, 1):
        print(f"  {i}. {strength}")
    
    weaknesses = review.get('weaknesses', [])
    print("\\nWeaknesses:")
    for i, weakness in enumerate(weaknesses, 1):
        print(f"  {i}. {weakness}")
    
    priority_improvements = review.get('priority_improvements', [])
    print("\\nPriority Improvements:")
    for i, improvement in enumerate(priority_improvements, 1):
        print(f"  {i}. {improvement}")
    
    if review.get('optimization_suggestions'):
        print("\\n" + "-" * 80)
        print("OPTIMIZATION SUGGESTIONS")
        print("-" * 80)
        print(f"\\n{review['optimization_suggestions']}")
    
    print("\\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
