# FROM_SCRATCH vs OPTIMIZATION: Side-by-Side Comparison

## Agent Comparison Table

| Aspect | FROM_SCRATCH Agent | OPTIMIZATION Agent |
|--------|-------------------|-------------------|
| **Primary Goal** | Create new letter from scratch | Revise existing letter based on feedback |
| **Input Requirements** | sender_detail + recipient_detail | sender_detail + recipient_detail + old_letter + feedback |
| **Constraints** | General SOP guidelines | Feedback points + original content bounds |
| **Freedom Level** | High (within factual bounds) | Constrained by feedback directives |
| **Prompt Skeleton** | `FROM_SCRATCH_PROMPT_SKELETON` | `OPTIMIZATION_PROMPT_SKELETON` |
| **Output Schema** | `LetterSchema` | `LetterSchema` (same) |
| **Cache Prefix** | `from_scratch.{key}.json` | `optimization.{key}.json` |
| **Typical Use** | First draft generation | Iterative improvement |
| **Main Focus** | Coverage & completeness | Addressing specific weaknesses |
| **Quality Metric** | Comprehensive SOP | Improvement over original |

---

## Workflow Comparison

### FROM_SCRATCH Workflow
```
Inputs:
├─ sender_detail (profile + resume)
├─ recipient_detail (program/lab info)
├─ tone (formal, informal, etc.)
└─ tailor_type (match_skills, etc.)
    │
    ▼
Generate pristine letter
├─ Opening (position + research interest)
├─ Experience (background + achievements)
├─ Fit (program alignment)
├─ Goals (career trajectory)
└─ Closing (summary + enthusiasm)
    │
    ▼
Output: Complete SOP letter
```

### OPTIMIZATION Workflow
```
Inputs:
├─ sender_detail (profile + resume)
├─ recipient_detail (program/lab info)
├─ old_letter (previous version)
├─ feedback (what to improve)
├─ revision_goals (objectives)
├─ tone (formal, informal, etc.)
└─ tailor_type (match_skills, etc.)
    │
    ▼
Analyze old letter
├─ Identify strengths
├─ Identify weaknesses
└─ Map to feedback points
    │
    ▼
Parse feedback
├─ Extract actionable changes
├─ Prioritize improvements
└─ Cross-reference profile
    │
    ▼
Revise strategically
├─ Address feedback point 1
├─ Address feedback point 2
├─ ...
└─ Preserve what works
    │
    ▼
Output: Improved SOP letter
```

---

## Prompt Structure Comparison

### FROM_SCRATCH Prompt Components
```
System Message:
┌─────────────────────────────────────┐
│ FROM_SCRATCH_PROMPT_SKELETON        │
│ - Expert academic writing agent     │
│ - Structure: Opening → Experience   │
│   → Fit → Goals → Closing           │
│ - Length: 400-800 tokens            │
│ - Quality checklist (7 points)      │
├─────────────────────────────────────┤
│ TONE_MODULE[tone]                   │
│ - Formal / Informal / Friendly      │
├─────────────────────────────────────┤
│ TAILOR_MODULES[types]               │
│ - match_skills / match_experience   │
├─────────────────────────────────────┤
│ AVOID / FOCUS                       │
│ - Custom constraints                │
├─────────────────────────────────────┤
│ DEFAULT_STYLE_ADDONS                │
│ - Sign-off preferences              │
└─────────────────────────────────────┘

User Message:
┌─────────────────────────────────────┐
│ <SENDER_DETAIL_JSON>                │
│ {...profile and resume...}          │
│ </SENDER_DETAIL_JSON>               │
├─────────────────────────────────────┤
│ <RECIPIENT_DETAIL_JSON>             │
│ {...program/lab info...}            │
│ </RECIPIENT_DETAIL_JSON>            │
└─────────────────────────────────────┘
```

### OPTIMIZATION Prompt Components
```
System Message:
┌─────────────────────────────────────┐
│ OPTIMIZATION_PROMPT_SKELETON        │
│ - Expert revision agent             │
│ - Honor feedback points             │
│ - Preserve factual accuracy         │
│ - Strategic revision workflow       │
│ - Quality checklist (8 points)      │
├─────────────────────────────────────┤
│ TONE_MODULE[tone]                   │
│ - Same as from-scratch              │
├─────────────────────────────────────┤
│ TAILOR_MODULES[types]               │
│ - Same as from-scratch              │
├─────────────────────────────────────┤
│ AVOID / FOCUS                       │
│ - Same as from-scratch              │
├─────────────────────────────────────┤
│ DEFAULT_STYLE_ADDONS                │
│ - Same as from-scratch              │
└─────────────────────────────────────┘

User Message:
┌─────────────────────────────────────┐
│ <SENDER_DETAIL_JSON>                │
│ {...profile and resume...}          │
│ </SENDER_DETAIL_JSON>               │
├─────────────────────────────────────┤
│ <RECIPIENT_DETAIL_JSON>             │
│ {...program/lab info...}            │
│ </RECIPIENT_DETAIL_JSON>            │
├─────────────────────────────────────┤
│ <OPTIMIZATION_CONTEXT_JSON>         │
│ {                                   │
│   "old_letter": {...},              │
│   "feedback": "...",                │
│   "revision_goals": [...]           │
│ }                                   │
│ </OPTIMIZATION_CONTEXT_JSON>        │
└─────────────────────────────────────┘
```

---

## Example Scenarios

### Scenario 1: First Time Writing

**User Need**: "I need to write an SOP for MIT CSAIL"

**Agent**: FROM_SCRATCH
```python
letter = await engine.generate(
    user_id="user123",
    sender_detail=alex_profile,
    recipient_detail=mit_info,
    tone="formal",
    generation_type="from_scratch"
)
```

**Output**: Complete SOP letter with opening, experience, fit, goals, closing

---

### Scenario 2: Improving Draft

**User Need**: "This SOP is too generic, add more details"

**Agent**: OPTIMIZATION
```python
letter_v2 = await engine.generate(
    user_id="user123",
    sender_detail=alex_profile,
    recipient_detail=mit_info,
    tone="formal",
    generation_type="optimization",
    optimization_context={
        "old_letter": letter,
        "feedback": "Add specific research projects and metrics",
        "revision_goals": ["add specifics"]
    }
)
```

**Output**: Improved SOP with specific details from resume

---

### Scenario 3: Iterative Refinement

**User Need**: "Make it shorter and more focused"

**Agent**: OPTIMIZATION (Round 2)
```python
letter_v3 = await engine.generate(
    user_id="user123",
    sender_detail=alex_profile,
    recipient_detail=mit_info,
    tone="formal",
    generation_type="optimization",
    optimization_context={
        "old_letter": letter_v2,
        "feedback": "Reduce length to fit one page",
        "revision_goals": ["reduce length", "maintain key points"]
    }
)
```

**Output**: Concise version maintaining essential information

---

## Prompt Skeleton Comparison

### FROM_SCRATCH Opening
```
You are AcademicScribe, an expert academic writing agent specializing in 
statements of purpose. Your job is to write a compelling, genuine, 
well-structured statement of purpose **in English** for graduate school, 
research positions, or academic programs, using only information from the 
applicant's profile and resume. Never fabricate facts.
```

### OPTIMIZATION Opening
```
You are AcademicScribe, an expert academic writing agent specializing in 
statement of purpose revision and optimization. Your job is to revise and 
improve an existing statement of purpose **in English** based on provided 
feedback and suggestions. You must maintain factual accuracy and only use 
information from the applicant's profile, resume, and the original letter.
```

**Key Difference**: FROM_SCRATCH creates; OPTIMIZATION revises based on feedback.

---

## Quality Checklist Comparison

### FROM_SCRATCH Checklist (7 points)
1. Recipient information is accurate and complete if provided
2. All claims are supported by information in the profile/resume
3. Technical skills and tools mentioned are present in the applicant's background
4. The narrative flows logically from background → fit → goals
5. Specific connections to the target program/lab are clear and genuine
6. Tone is appropriately academic yet personal
7. Body is well-structured with clear transitions between paragraphs

### OPTIMIZATION Checklist (8 points)
1. **All feedback points have been addressed appropriately** ← New
2. No new facts, skills, or experiences have been added that aren't in the profile/resume/original letter
3. **The revised letter is more compelling, clear, or effective than the original** ← New
4. Recipient and signature information remains accurate and complete
5. Body maintains proper academic tone and structure
6. All LaTeX special characters are properly escaped
7. **Length and structural constraints from revision_goals (if any) are met** ← New
8. The narrative flows logically from background → fit → goals

**Key Additions**: Feedback coverage, improvement verification, constraint compliance

---

## When to Use Which Agent

### Use FROM_SCRATCH When:
- ✅ Creating the first draft
- ✅ No existing letter to work from
- ✅ User wants a completely fresh perspective
- ✅ Starting a new application
- ✅ Experimenting with different approaches

### Use OPTIMIZATION When:
- ✅ Already have a draft (generated or human-written)
- ✅ Specific improvements needed
- ✅ User provided feedback
- ✅ Iterative refinement process
- ✅ Making targeted changes without full rewrite

### Consider Both (Sequential) When:
- ✅ Generate initial draft with FROM_SCRATCH
- ✅ User reviews and provides feedback
- ✅ Optimize with OPTIMIZATION based on feedback
- ✅ Repeat optimization as needed
- ✅ Final polish before submission

---

## Code Paths

### FROM_SCRATCH Path
```python
# In engine.py:99-148 (build_messages)
if action_type == "from_scratch":
    prompt_skeleton = FROM_SCRATCH_PROMPT_SKELETON
    # No optimization_context needed
    contexts = {
        "SENDER_DETAIL_JSON": ...,
        "RECIPIENT_DETAIL_JSON": ...
    }
```

### OPTIMIZATION Path
```python
# In engine.py:99-148 (build_messages)
if action_type == "optimization":
    prompt_skeleton = OPTIMIZATION_PROMPT_SKELETON
    # Add optimization_context
    contexts = {
        "SENDER_DETAIL_JSON": ...,
        "RECIPIENT_DETAIL_JSON": ...,
        "OPTIMIZATION_CONTEXT_JSON": ...  # Additional input
    }
```

---

## Performance Considerations

| Aspect | FROM_SCRATCH | OPTIMIZATION |
|--------|--------------|--------------|
| **Token Count** | Medium (~800-1200) | Higher (~1200-2000+) |
| **Latency** | Standard | Slightly higher |
| **Cache Hit Rate** | Lower (unique combos) | Higher (same letter + feedback) |
| **Cost** | Standard | 20-50% higher (more tokens) |
| **Quality Variance** | Moderate | Lower (constrained by feedback) |

---

## Summary

Both agents work **harmoniously** together:

1. **FROM_SCRATCH** creates the initial draft
2. **OPTIMIZATION** refines based on feedback
3. They share:
   - Same output schema
   - Same rendering pipeline
   - Same tone/tailor modules
   - Same quality standards
4. They differ in:
   - Input requirements
   - Prompt skeleton
   - Primary objective
   - Constraint model

This two-agent system enables a **complete lifecycle** for academic letter generation:
- **Create** with FROM_SCRATCH
- **Improve** with OPTIMIZATION
- **Iterate** until perfect
