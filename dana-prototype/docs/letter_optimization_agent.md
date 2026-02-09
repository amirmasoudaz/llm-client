# Letter Optimization Agent - Prompt Skeleton Documentation

## Overview
The `OPTIMIZATION_PROMPT_SKELETON` is now fully implemented in `/home/namiral/Projects/CanApply/dana-funding/src/agents/letter/context.py` (lines 57-118).

## Architecture

### Input Structure
The optimization agent receives three inputs via the `optimization_context` parameter:

```json
{
  "old_letter": "Original letter text or structured JSON with all fields",
  "feedback": "Specific suggestions, critiques, or improvement requests",
  "revision_goals": ["List of objectives like 'strengthen research fit'"]
}
```

This context is automatically injected as `OPTIMIZATION_CONTEXT_JSON` in the user message when `action_type="optimization"` (see `engine.py` lines 137-138).

### Output Structure
The agent outputs the same `LetterSchema` structure as the from-scratch generation:
- `recipient_name`, `recipient_position`, `recipient_institution`, `recipient_city`, `recipient_country`
- `signature_name`, `signature_city`, `signature_country`, `signature_phone`, `signature_email`, `signature_linkedin`
- `date`, `salutation`, `body`, `closing_valediction`

## Key Design Principles

### 1. **Preserve Accuracy**
The agent is explicitly instructed to **never fabricate facts**. It can only use information from:
- The applicant's profile (in `sender_detail`)
- The applicant's resume (in `sender_detail`)
- The original letter (in `optimization_context.old_letter`)

### 2. **Honor Feedback**
The agent must directly address all feedback points provided in `optimization_context.feedback`. This is the primary driver of the revision.

### 3. **Strategic Revision Workflow**
The prompt includes specific transformation patterns:
- **Vague → Specific**: Add concrete details from the resume
- **Generic → Tailored**: Connect to specific research areas
- **Weak → Strong**: Use evidence-based language
- **Redundant → Concise**: Remove repetition
- **Scattered → Coherent**: Improve logical flow

### 4. **Quality Assurance**
The prompt includes an 8-point quality checklist that the agent performs before generating output.

## Usage Example

```python
from src.agents.letter.engine import LetterEngine

engine = LetterEngine()

optimized_letter = await engine.generate(
    user_id="user123",
    sender_detail={
        "identity": {...},
        "profile": {...},
        "resume": {...}
    },
    recipient_detail={
        "name": "Dr. Jane Smith",
        "institution": "MIT CSAIL",
        ...
    },
    tone="formal",
    tailor_type=["match_skills", "match_experience"],
    generation_type="optimization",  # Key parameter
    optimization_context={
        "old_letter": previous_letter_json_or_text,
        "feedback": "The opening is too generic. Add more specific details about your research experience with neural networks.",
        "revision_goals": ["strengthen research fit", "add technical depth"]
    }
)
```

## How It Works

1. **Message Building** (`engine.py:99-148`):
   - When `action_type="optimization"`, the system uses `OPTIMIZATION_PROMPT_SKELETON` instead of `FROM_SCRATCH_PROMPT_SKELETON`
   - The `optimization_context` is serialized as JSON and injected into the user message as `<OPTIMIZATION_CONTEXT_JSON>`

2. **Prompt Construction**:
   - System message: `OPTIMIZATION_PROMPT_SKELETON` + tone modules + tailor modules + avoid/focus + style add-ons
   - User message: `SENDER_DETAIL_JSON` + `RECIPIENT_DETAIL_JSON` + `OPTIMIZATION_CONTEXT_JSON`

3. **Agent Processing**:
   - The LLM receives the optimization instructions and context
   - Analyzes the old letter against the feedback
   - Revises strategically while preserving factual accuracy
   - Outputs a new `LetterSchema` with improved content

4. **Post-Processing** (`engine.py:198-220`):
   - Unicode normalization for LaTeX
   - Valediction cleaning
   - Schema validation

## Comparison with FROM_SCRATCH

| Aspect | FROM_SCRATCH | OPTIMIZATION |
|--------|--------------|--------------|
| **Input** | Profile + recipient only | Profile + recipient + old letter + feedback |
| **Task** | Create new letter from zero | Revise existing letter based on feedback |
| **Constraints** | Follow general SOP guidelines | Honor feedback + preserve accuracy |
| **Freedom** | High creativity within guidelines | Constrained by feedback and original content |
| **Use Case** | First draft generation | Iterative improvement |

## Common Optimization Scenarios

The prompt includes specific guidance for common revision requests:

- **"Strengthen research fit"** → Add connections between applicant's work and target lab
- **"Too generic"** → Incorporate concrete project names and methodologies
- **"Weak opening"** → Revise first paragraph to be more compelling
- **"Reduce length"** → Remove redundancy and tighten language
- **"Add technical depth"** → Integrate specific tools and frameworks

## Next Steps

To use this optimization agent in production:

1. **Frontend Integration**: Add UI for users to provide feedback on generated letters
2. **Feedback Templates**: Create structured feedback templates for common issues
3. **Iterative Refinement**: Allow multiple optimization passes
4. **Feedback History**: Track feedback and revisions for learning
5. **A/B Testing**: Compare before/after versions to measure improvement

## Notes

- The optimization agent shares the same `LetterSchema` output format as from-scratch generation
- It can be combined with all existing tone and tailoring options
- Cache keys are separate for `from_scratch` and `optimization` generations (see `engine.py:303`)
- The same rendering pipeline works for both types of letters
