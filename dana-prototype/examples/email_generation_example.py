"""
Example script demonstrating email generation from scratch using the EmailEngine.

This script generates a professor outreach email based on sender and recipient details.
"""

import asyncio
from src.agents.email.engine import EmailEngine


async def main():
    engine = EmailEngine()
    
    # Sender details (student/researcher reaching out)
    sender_detail = {
        "identity": {
            "full_name": "Jane Doe",
            "email": "jane.doe@university.edu",
            "phone": "+1-555-123-4567"
        },
        "current_status": "PhD student in Computer Science at State University",
        "research_interests": ["machine learning", "computer vision", "neural architecture search"],
        "background": "3 years of research experience in deep learning",
        "achievements": [
            "Published 2 papers in CVPR 2024 on CNN optimization",
            "Developed automated neural architecture search tool achieving 15% speedup",
            "First-author publication on efficient vision transformers"
        ],
        "skills": ["PyTorch", "TensorFlow", "Python", "CUDA", "Docker"],
        "projects": [
            {
                "name": "AutoVision",
                "description": "Automated CNN architecture search for medical imaging",
                "outcome": "94% accuracy on BraTS dataset, 40% faster than baseline"
            }
        ]
    }
    
    # Recipient details (professor being contacted)
    recipient_detail = {
        "name": "Dr. John Smith",
        "position": "Professor",
        "institution": "MIT CSAIL",
        "department": "Computer Science and Artificial Intelligence Laboratory",
        "research_areas": ["deep learning", "neural architecture search", "AutoML", "efficient deep learning"],
        "recent_work": "Recent NeurIPS paper on differentiable architecture search and AutoML for edge devices",
        "lab_culture": "Emphasis on open-source contributions and interdisciplinary collaboration"
    }
    
    # Generate email from scratch
    print("Generating professor outreach email...\\n")
    
    email = await engine.generate(
        user_id="test_user_001",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",
        tailor_type=["match_research_area", "match_recent_papers"],
        cache=False  # Set to False for testing to avoid caching
    )
    
    # Display the generated email
    print("=" * 80)
    print("GENERATED EMAIL")
    print("=" * 80)
    print(f"\\nSubject: {email.get('subject', 'N/A')}")
    print(f"\\nGreeting: {email.get('greeting', 'N/A')}")
    print(f"\\nBody:\\n{email.get('body', 'N/A')}")
    print(f"\\nClosing: {email.get('closing', 'N/A')}")
    print(f"\\nSignature:")
    print(f"  {email.get('signature_name', 'N/A')}")
    print(f"  {email.get('signature_email', 'N/A')}")
    if email.get('signature_phone'):
        print(f"  {email['signature_phone']}")
    if email.get('signature_linkedin'):
        print(f"  LinkedIn: {email['signature_linkedin']}")
    print("=" * 80)
    
    # Optional: Generate HTML version
    html = engine.render_html(email)
    html_path = "/tmp/generated_email.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\\nHTML version saved to: {html_path}")
    print("You can open this file in a browser to see the formatted email.\\n")


if __name__ == "__main__":
    asyncio.run(main())
