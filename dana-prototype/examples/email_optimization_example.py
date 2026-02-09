"""
Example script demonstrating email optimization/regeneration using the EmailEngine.

This script takes a weak email and improves it based on specific feedback.
"""

import asyncio
from src.agents.email.engine import EmailEngine


async def main():
    engine = EmailEngine()
    
    # Sender details
    sender_detail = {
        "identity": {
            "full_name": "Jane Doe",
            "email": "jane.doe@university.edu",
            "phone": "+1-555-123-4567"
        },
        "current_status": "PhD student in Computer Science",
        "research_interests": ["computer vision", "neural networks", "AutoML"],
        "achievements": [
            "Published paper on CNN optimization in CVPR 2024",
            "Developed AutoVision tool achieving 94% accuracy on medical imaging"
        ],
        "skills": ["PyTorch", "TensorFlow", "Python"]
    }
    
    # Recipient details
    recipient_detail = {
        "name": "Dr. John Smith",
        "position": "Professor",
        "institution": "MIT CSAIL",
        "research_areas": ["deep learning", "AutoML", "neural architecture search"],
        "recent_work": "NeurIPS 2024 paper on differentiable architecture search"
    }
    
    # Original weak email
    old_email = {
        "subject": "Research Collaboration",
        "greeting": "Dear Dr. Smith,",
        "body": "I am interested in your work on AutoML. I have experience in neural networks and have done some projects. "
                "I would like to collaborate with you.",
        "closing": "Thanks,",
        "signature_name": "Jane Doe",
        "signature_email": "jane.doe@university.edu"
    }
    
    # Feedback for optimization
    optimization_context = {
        "old_email": old_email,
        "feedback": (
            "Subject line is too generic and doesn't specify research area. "
            "Body lacks concrete examples and specific achievements. "
            "No connection to professor's specific work (e.g., recent NeurIPS paper on differentiable architecture search). "
            "Missing clear call to action with timeframe. "
            "Overall too brief and vague - needs specific project details and metrics."
        ),
        "revision_goals": [
            "Make subject line specific with research area",
            "Add concrete achievements with metrics (CVPR paper, 94% accuracy)",
            "Reference professor's NeurIPS paper explicitly",
            "Add clear call to action with specific timeframe",
            "Target 200-250 words for body"
        ]
    }
    
    print("ORIGINAL EMAIL:")
    print("=" * 80)
    print(f"Subject: {old_email['subject']}")
    print(f"\\n{old_email['greeting']}")
    print(f"\\n{old_email['body']}")
    print(f"\\n{old_email['closing']}")
    print(f"{old_email['signature_name']}")
    print("=" * 80)
    
    print("\\nOptimizing email based on feedback...\\n")
    
    # Generate optimized email
    improved_email = await engine.generate(
        user_id="test_user_001",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",
        tailor_type=["match_research_area", "match_recent_papers"],
        generation_type="optimization",
        optimization_context=optimization_context,
        cache=False  # Set to False for testing
    )
    
    print("OPTIMIZED EMAIL:")
    print("=" * 80)
    print(f"Subject: {improved_email.get('subject', 'N/A')}")
    print(f"\\n{improved_email.get('greeting', 'N/A')}")
    print(f"\\n{improved_email.get('body', 'N/A')}")
    print(f"\\n{improved_email.get('closing', 'N/A')}")
    print(f"{improved_email.get('signature_name', 'N/A')}")
    if improved_email.get('signature_email'):
        print(improved_email['signature_email'])
    if improved_email.get('signature_phone'):
        print(improved_email['signature_phone'])
    print("=" * 80)
    
    print("\\nKEY IMPROVEMENTS:")
    print("-" * 80)
    print(f"1. Subject: '{old_email['subject']}' → '{improved_email.get('subject', 'N/A')}'")
    print(f"2. Body length: ~{len(old_email['body'].split())} words → ~{len(improved_email.get('body', '').split())} words")
    print("3. Added specific achievements and metrics")
    print("4. Referenced professor's specific research")
    print("5. Added clear call to action")
    print("-" * 80)
    
    # Optional: Generate HTML versions for comparison
    old_html = engine.render_html(old_email)
    new_html = engine.render_html(improved_email)
    
    with open("/tmp/email_old.html", "w", encoding="utf-8") as f:
        f.write(old_html)
    with open("/tmp/email_improved.html", "w", encoding="utf-8") as f:
        f.write(new_html)
    
    print("\\nHTML versions saved to:")
    print("  - /tmp/email_old.html")
    print("  - /tmp/email_improved.html")
    print("Open these files in a browser to compare side-by-side.\\n")


if __name__ == "__main__":
    asyncio.run(main())
