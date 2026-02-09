"""
Letter Optimization Agent - Usage Examples
==========================================

This file demonstrates how to use the letter optimization agent to improve
existing statement of purpose letters based on feedback.
"""

import asyncio
from src.agents.letter.engine import LetterEngine

# Example 1: Basic Optimization
async def basic_optimization_example():
    """
    Optimize a letter with simple feedback.
    """
    engine = LetterEngine()
    
    # Original letter (from previous generation)
    old_letter = {
        "recipient_name": "Dr. Sarah Johnson",
        "recipient_position": "Associate Professor",
        "recipient_institution": "MIT Computer Science and Artificial Intelligence Laboratory",
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
        "body": "I am writing to express my interest in your PhD program. I have a background in machine learning and computer vision. I believe your program would be a great fit for me.",
        "closing_valediction": "Sincerely"
    }
    
    # Sender details (profile + resume)
    sender_detail = {
        "identity": {
            "full_name": "Alex Chen",
            "email": "alex.chen@example.com",
            "phone": "+1-416-555-0123",
            "city": "Toronto",
            "country": "Canada",
            "linkedin": "https://linkedin.com/in/alexchen"
        },
        "education": [
            {
                "degree": "M.Sc. Computer Science",
                "institution": "University of Toronto",
                "year": "2023",
                "thesis": "Attention Mechanisms in Medical Image Segmentation"
            }
        ],
        "research_experience": [
            {
                "title": "Graduate Researcher",
                "lab": "Vision and Image Processing Lab",
                "advisor": "Dr. Maria Rodriguez",
                "duration": "2021-2023",
                "description": "Developed a novel attention-based neural network for brain tumor segmentation in MRI scans",
                "achievements": [
                    "Published paper at MICCAI 2023",
                    "Achieved 95% Dice coefficient on BraTS dataset",
                    "Reduced inference time by 40% compared to baseline"
                ],
                "tools": ["PyTorch", "MONAI", "Docker", "AWS"]
            }
        ],
        "publications": [
            {
                "title": "Efficient Attention Mechanisms for 3D Medical Image Segmentation",
                "venue": "MICCAI 2023",
                "authors": "A. Chen, M. Rodriguez",
                "status": "published"
            }
        ]
    }
    
    recipient_detail = {
        "name": "Dr. Sarah Johnson",
        "position": "Associate Professor",
        "institution": "MIT CSAIL",
        "department": "Computer Science",
        "research_areas": ["Deep Learning for Healthcare", "Medical Image Analysis", "Interpretable AI"],
        "city": "Cambridge, MA",
        "country": "USA"
    }
    
    # Optimization context with feedback
    optimization_context = {
        "old_letter": old_letter,
        "feedback": """
        The letter is too generic and lacks specific details. Please:
        1. Mention your specific research project on brain tumor segmentation
        2. Connect your work to Dr. Johnson's research on medical image analysis
        3. Include the publication at MICCAI 2023
        4. Add technical details about the attention mechanism you developed
        5. Make the opening more compelling and specific
        """,
        "revision_goals": [
            "strengthen research fit",
            "add technical depth",
            "improve opening paragraph"
        ]
    }
    
    # Call the optimization agent
    optimized_letter = await engine.generate(
        user_id="alex_chen_123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",
        tailor_type=["match_skills", "match_experience"],
        generation_type="optimization",
        optimization_context=optimization_context,
        cache=True
    )
    
    print("=== OPTIMIZED LETTER ===")
    print(f"Body:\n{optimized_letter['body']}\n")
    return optimized_letter


# Example 2: Iterative Optimization
async def iterative_optimization_example():
    """
    Perform multiple rounds of optimization with different feedback.
    """
    engine = LetterEngine()
    
    # Simplified sender/recipient for brevity
    sender_detail = {...}  # Same as above
    recipient_detail = {...}  # Same as above
    
    # Round 1: Address initial feedback
    round1_context = {
        "old_letter": original_letter,
        "feedback": "Too generic. Add specific research details.",
        "revision_goals": ["add specifics"]
    }
    
    letter_v1 = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context=round1_context
    )
    
    # Round 2: Further refinement
    round2_context = {
        "old_letter": letter_v1,
        "feedback": "Good progress. Now reduce the length to fit one page.",
        "revision_goals": ["reduce length", "maintain key points"]
    }
    
    letter_v2 = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context=round2_context
    )
    
    # Round 3: Final polish
    round3_context = {
        "old_letter": letter_v2,
        "feedback": "Add a stronger closing that emphasizes long-term career goals.",
        "revision_goals": ["strengthen closing"]
    }
    
    final_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context=round3_context
    )
    
    return final_letter


# Example 3: Structured Feedback
async def structured_feedback_example():
    """
    Use structured feedback for more precise control.
    """
    engine = LetterEngine()
    
    sender_detail = {...}
    recipient_detail = {...}
    old_letter = {...}
    
    # Structured feedback by section
    optimization_context = {
        "old_letter": old_letter,
        "feedback": {
            "opening": "Too generic. Start with your MICCAI publication and its relevance to Dr. Johnson's work.",
            "experience": "Include specific metrics from your thesis work (95% Dice coefficient, 40% speedup).",
            "fit": "Explicitly mention Dr. Johnson's recent papers on interpretable AI in healthcare.",
            "closing": "Express specific interest in contributing to the lab's current projects."
        },
        "revision_goals": [
            "strengthen research fit",
            "add quantitative achievements",
            "improve specificity throughout"
        ]
    }
    
    optimized_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",
        generation_type="optimization",
        optimization_context=optimization_context
    )
    
    return optimized_letter


# Example 4: Addressing Specific Weaknesses
async def address_weaknesses_example():
    """
    Target specific weaknesses identified in the original letter.
    """
    engine = LetterEngine()
    
    sender_detail = {...}
    recipient_detail = {...}
    old_letter = {...}
    
    # Focus on common weaknesses
    optimization_context = {
        "old_letter": old_letter,
        "feedback": """
        Identified weaknesses:
        - VAGUE: "I have experience in machine learning" is too broad
        - MISSING: No mention of PyTorch, MONAI, or Docker skills
        - WEAK: "Would be a great fit" needs evidence-based justification
        - REPETITIVE: "Research" appears 8 times; vary the language
        
        Action items:
        - Replace vague statements with concrete project details
        - Add technical stack alignment (PyTorch, medical imaging libraries)
        - Support "fit" claim with specific research area overlaps
        - Improve vocabulary diversity
        """,
        "revision_goals": [
            "eliminate vagueness",
            "add technical skills",
            "strengthen justification",
            "improve writing quality"
        ]
    }
    
    optimized_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context=optimization_context,
        cache=False  # Don't cache during experimentation
    )
    
    return optimized_letter


# Example 5: Tone and Style Adjustment
async def tone_adjustment_example():
    """
    Adjust tone while keeping content factual.
    """
    engine = LetterEngine()
    
    sender_detail = {...}
    recipient_detail = {...}
    old_letter = {...}
    
    # Adjust from overly casual to appropriately formal
    optimization_context = {
        "old_letter": old_letter,
        "feedback": """
        The tone is too casual for an academic SOP. Please:
        - Remove contractions ("I'm" → "I am")
        - Replace informal phrases ("really excited" → "deeply interested")
        - Maintain professional academic voice throughout
        - Keep the content authentic but elevate the language register
        """,
        "revision_goals": [
            "adjust tone to formal academic",
            "maintain authenticity"
        ]
    }
    
    optimized_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        tone="formal",  # Reinforce formal tone
        generation_type="optimization",
        optimization_context=optimization_context
    )
    
    return optimized_letter


# Example 6: Full Integration with Rendering
async def full_pipeline_example():
    """
    Complete workflow: optimize and render to PDF.
    """
    engine = LetterEngine()
    
    sender_detail = {...}
    recipient_detail = {...}
    old_letter = {...}
    
    # Step 1: Optimize
    optimization_context = {
        "old_letter": old_letter,
        "feedback": "Strengthen technical details and research fit.",
        "revision_goals": ["add technical depth", "improve fit"]
    }
    
    optimized_letter = await engine.generate(
        user_id="user123",
        sender_detail=sender_detail,
        recipient_detail=recipient_detail,
        generation_type="optimization",
        optimization_context=optimization_context
    )
    
    # Step 2: Render to PDF
    rendered = await engine.render(
        letter=optimized_letter,
        letter_type="sop",
        compile_pdf=True,
        margin=1.0,
        min_margin=0.5
    )
    
    print(f"Optimized letter compiled to: {rendered['pdf_path']}")
    print(f"Page count: {rendered.get('page_count', 'unknown')}")
    print(f"Compile status: {rendered['compile_status']}")
    
    return rendered


# Run examples
if __name__ == "__main__":
    # Run the basic example
    asyncio.run(basic_optimization_example())
    
    # To run other examples:
    # asyncio.run(iterative_optimization_example())
    # asyncio.run(structured_feedback_example())
    # asyncio.run(address_weaknesses_example())
    # asyncio.run(tone_adjustment_example())
    # asyncio.run(full_pipeline_example())
