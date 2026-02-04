```mermaid
graph TD
    A[User Input] -->|sender_detail, recipient_detail| B[LetterEngine.generate]
    A -->|generation_type='optimization'| B
    A -->|optimization_context| B
    
    B --> C{Check generation_type}
    C -->|from_scratch| D[FROM_SCRATCH_PROMPT_SKELETON]
    C -->|optimization| E[OPTIMIZATION_PROMPT_SKELETON]
    
    D --> F[Build Messages]
    E --> F
    
    F --> G[System Prompt]
    F --> H[User Payload]
    
    G --> I[Prompt Skeleton]
    G --> J[Tone Modules]
    G --> K[Tailor Modules]
    G --> L[Avoid/Focus]
    G --> M[Style Add-ons]
    
    H --> N[SENDER_DETAIL_JSON]
    H --> O[RECIPIENT_DETAIL_JSON]
    H --> P{optimization?}
    P -->|yes| Q[OPTIMIZATION_CONTEXT_JSON]
    P -->|no| R[Skip]
    
    Q --> S[old_letter]
    Q --> T[feedback]
    Q --> U[revision_goals]
    
    I --> V[OpenAI API]
    J --> V
    K --> V
    L --> V
    M --> V
    N --> V
    O --> V
    S --> V
    T --> V
    U --> V
    
    V --> W[LLM Response]
    W --> X[LetterSchema Validation]
    X --> Y[Post-Validation]
    Y --> Z[Unicode Normalization]
    Y --> AA[Valediction Cleaning]
    
    Z --> AB[Optimized Letter JSON]
    AA --> AB
    
    AB --> AC[Cache to Disk]
    AB --> AD[Return to Caller]
    
    style E fill:#ff9999
    style Q fill:#ff9999
    style S fill:#ffcccc
    style T fill:#ffcccc
    style U fill:#ffcccc
```

## Optimization Context Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION CONTEXT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  old_letter: {                                                  │
│    "recipient_name": "Dr. Jane Smith",                          │
│    "body": "I am writing to express my interest...",            │
│    ...                                                          │
│  }                                                              │
│                                                                 │
│  feedback: "The opening paragraph is too generic. Mention       │
│             specific research projects from your resume that    │
│             align with Prof. Smith's work on neural networks."  │
│                                                                 │
│  revision_goals: [                                              │
│    "strengthen research fit",                                   │
│    "add technical depth"                                        │
│  ]                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│               OPTIMIZATION_PROMPT_SKELETON                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Analyze old_letter                                          │
│  2. Parse feedback → "generic opening"                          │
│  3. Cross-reference sender_detail (profile/resume)              │
│  4. Find: "CNN project", "90% accuracy", "medical imaging"      │
│  5. Rewrite strategically:                                      │
│     - Replace generic → specific research project               │
│     - Add connection to Prof. Smith's neural network work       │
│  6. Validate: no fabrication, feedback addressed                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROCESSING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BEFORE:                                                        │
│  "I am writing to express my interest in your program..."       │
│                                                                 │
│  AFTER:                                                         │
│  "I am writing to express my interest in joining your           │
│   research group, as your work on convolutional neural          │
│   networks for medical applications directly aligns with        │
│   my thesis project, where I developed a CNN-based classifier   │
│   that achieved 94% accuracy on breast cancer detection..."     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         OPTIMIZED LETTER
```

## Revision Transformation Patterns

```
┌──────────────────────┐          ┌──────────────────────┐
│   FEEDBACK TYPE      │          │   AGENT ACTION       │
├──────────────────────┤          ├──────────────────────┤
│                      │          │                      │
│ "Too vague"          │─────────▶│ Add specifics from   │
│                      │          │ resume/profile       │
│                      │          │                      │
│ "Generic"            │─────────▶│ Tailor to target     │
│                      │          │ lab/program          │
│                      │          │                      │
│ "Weak evidence"      │─────────▶│ Add concrete         │
│                      │          │ metrics/outcomes     │
│                      │          │                      │
│ "Too long"           │─────────▶│ Remove redundancy,   │
│                      │          │ consolidate points   │
│                      │          │                      │
│ "Missing technical   │─────────▶│ Add tools/methods    │
│  depth"              │          │ from background      │
│                      │          │                      │
│ "Poor structure"     │─────────▶│ Reorganize for       │
│                      │          │ logical flow         │
│                      │          │                      │
└──────────────────────┘          └──────────────────────┘
```
