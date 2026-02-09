# Letter Optimization Context - Quick Reference

## Structure

The `optimization_context` parameter accepts a dictionary with the following structure:

```python
optimization_context = {
    "old_letter": dict | str,           # Required
    "feedback": str | dict,              # Required
    "revision_goals": list[str]          # Optional
}
```

---

## 1. `old_letter` (Required)

The original letter to be optimized. Can be provided in two formats:

### Format A: Full LetterSchema Dictionary
```python
"old_letter": {
    "recipient_name": "Dr. Jane Smith",
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
    "salutation": "Dear Dr. Smith,",
    "body": "Full letter body text here...",
    "closing_valediction": "Sincerely"
}
```

### Format B: Plain Text
```python
"old_letter": """
Dear Dr. Smith,

I am writing to express my interest in your PhD program...

[Full letter text]

Sincerely,
Alex Chen
"""
```

**Note**: Format A is preferred as it preserves all structured fields.

---

## 2. `feedback` (Required)

Instructions for how to improve the letter. Can be provided as:

### Format A: Simple String
```python
"feedback": "The opening paragraph is too generic. Add specific details about your research project on neural networks."
```

### Format B: Multi-line String
```python
"feedback": """
Please address the following issues:
1. The opening is too vague - mention your MICCAI publication
2. Add technical details about your attention mechanism
3. Connect your work to Dr. Smith's research on medical imaging
4. Strengthen the closing paragraph with specific career goals
"""
```

### Format C: Structured Dictionary (Recommended)
```python
"feedback": {
    "opening": "Too generic. Start with your MICCAI publication.",
    "experience": "Include the 95% Dice coefficient metric from your thesis.",
    "fit": "Mention Dr. Smith's recent paper on interpretable AI.",
    "closing": "Add specific interest in the lab's current projects.",
    "tone": "Maintain formal academic tone throughout.",
    "length": "Reduce by ~100 words to fit one page."
}
```

---

## 3. `revision_goals` (Optional)

High-level objectives for the optimization. Common goals:

```python
"revision_goals": [
    # Content improvements
    "strengthen research fit",
    "add technical depth",
    "improve specificity",
    "add quantitative achievements",
    "connect to target lab",
    
    # Structure improvements
    "improve opening paragraph",
    "strengthen closing",
    "improve logical flow",
    "better paragraph transitions",
    
    # Style improvements
    "reduce length",
    "tighten language",
    "eliminate redundancy",
    "improve vocabulary diversity",
    "adjust tone to formal",
    
    # Evidence improvements
    "add concrete examples",
    "include metrics",
    "strengthen evidence",
    "support claims with facts"
]
```

---

## Complete Examples

### Example 1: Basic Optimization
```python
optimization_context = {
    "old_letter": previous_letter_dict,
    "feedback": "Add more specific details about your research experience.",
    "revision_goals": ["add specifics", "strengthen evidence"]
}
```

### Example 2: Targeted Improvements
```python
optimization_context = {
    "old_letter": previous_letter_dict,
    "feedback": {
        "opening": "Mention your publication at CVPR 2024",
        "experience": "Add the 40% speed improvement metric",
        "fit": "Connect your GAN work to Prof. Lee's research"
    },
    "revision_goals": [
        "strengthen research fit",
        "add quantitative achievements"
    ]
}
```

### Example 3: Length Reduction
```python
optimization_context = {
    "old_letter": previous_letter_dict,
    "feedback": "Too long. Reduce to fit one page while keeping key achievements.",
    "revision_goals": [
        "reduce length",
        "eliminate redundancy",
        "maintain key points"
    ]
}
```

### Example 4: Tone Adjustment
```python
optimization_context = {
    "old_letter": previous_letter_dict,
    "feedback": "The tone is too casual. Make it more formal and academic.",
    "revision_goals": [
        "adjust tone to formal academic",
        "maintain authenticity"
    ]
}
```

### Example 5: Comprehensive Revision
```python
optimization_context = {
    "old_letter": previous_letter_dict,
    "feedback": """
    Major issues:
    - Opening paragraph lacks specificity
    - Missing mention of your 2 publications
    - No connection to Prof. Wang's research areas
    - Technical skills (PyTorch, TensorFlow) not mentioned
    - Closing is weak
    
    Please revise to address these gaps while keeping content truthful.
    """,
    "revision_goals": [
        "strengthen research fit",
        "add technical depth",
        "improve opening and closing",
        "include publications"
    ]
}
```

---

## Common Feedback Patterns

### Pattern 1: Too Vague → Add Specifics
```python
"feedback": "Replace 'I have experience in machine learning' with specific projects, tools, and outcomes from your resume."
```

### Pattern 2: Too Generic → Add Tailoring
```python
"feedback": "Connect your CNN work on medical imaging to Dr. Johnson's recent MICCAI papers on the same topic."
```

### Pattern 3: Weak Evidence → Add Metrics
```python
"feedback": "Add quantitative results: 95% accuracy, 40% speedup, published at ICCV 2024."
```

### Pattern 4: Poor Structure → Reorganize
```python
"feedback": "Move the discussion of your thesis to paragraph 2, right after the opening. Save the future goals for the final paragraph."
```

### Pattern 5: Missing Skills → Add Technical Details
```python
"feedback": "Mention your proficiency with PyTorch, MONAI, Docker, and AWS from your research experience."
```

---

## Best Practices

### ✅ Do:
- Be specific about what needs to change
- Reference concrete items from the applicant's profile/resume
- Provide clear, actionable feedback
- Focus on 2-3 major improvements per round
- Use structured feedback for complex revisions

### ❌ Don't:
- Ask the agent to fabricate information
- Request changes that contradict the applicant's background
- Provide vague feedback like "make it better"
- Request massive changes in a single optimization pass
- Ignore the constraint of factual accuracy

---

## Iterative Optimization Strategy

For best results, optimize in multiple passes:

**Round 1**: Core content improvements
```python
{"feedback": "Add specific research details and technical depth"}
```

**Round 2**: Structure and flow
```python
{"feedback": "Improve paragraph transitions and logical flow"}
```

**Round 3**: Length and polish
```python
{"feedback": "Reduce length by 15% while keeping key achievements"}
```

**Round 4**: Final refinements
```python
{"feedback": "Strengthen opening hook and closing statement"}
```

---

## Testing Your Optimization

To verify the optimization worked:

1. **Check Feedback Addressed**: All points in feedback should be reflected in the new letter
2. **Verify Factual Accuracy**: All claims should be traceable to profile/resume/old letter
3. **Assess Improvement**: New letter should be measurably better than old letter
4. **Test Rendering**: Compile to PDF and check page count, layout, formatting
5. **Human Review**: Final check for tone, coherence, and appropriateness

---

## Common Use Cases

| Use Case | Feedback Example | Revision Goals |
|----------|------------------|----------------|
| First draft too generic | "Add specific research projects and outcomes" | `["add specifics", "strengthen evidence"]` |
| Missing technical skills | "Mention PyTorch, MONAI, Docker from resume" | `["add technical depth"]` |
| Poor research fit | "Connect your work to Prof. X's papers" | `["strengthen research fit"]` |
| Too long | "Reduce to fit one page" | `["reduce length", "eliminate redundancy"]` |
| Weak opening | "Start with your CVPR publication" | `["improve opening"]` |
| Lacks metrics | "Add 95% accuracy and 40% speedup" | `["add quantitative achievements"]` |
| Tone issues | "More formal and academic" | `["adjust tone to formal"]` |

---

## Integration Points

The optimization context integrates with:

- **LetterEngine.generate()**: Pass as `optimization_context` parameter
- **Tone modules**: Works with all tone options (formal, informal, friendly, enthusiastic)
- **Tailor modules**: Compatible with match_skills, match_experience, etc.
- **Avoid/Focus**: Can be combined with avoid and focus parameters
- **Caching**: Separate cache keys for optimization vs from_scratch

---

## Error Handling

If optimization fails, check:

1. **Missing old_letter**: Must provide the original letter
2. **Empty feedback**: Feedback cannot be empty or None
3. **Conflicting goals**: Avoid contradictory revision_goals
4. **Invalid letter format**: old_letter should match LetterSchema or be plain text
5. **Profile mismatch**: Feedback shouldn't reference skills not in sender_detail
