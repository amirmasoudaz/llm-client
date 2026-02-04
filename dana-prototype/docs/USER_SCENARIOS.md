# Dana AI Copilot - User Scenarios & Edge Cases

> Complete coverage of user actions, behaviors, and scenarios with edge cases and next steps

## Table of Contents

1. [Thread Lifecycle](#1-thread-lifecycle)
2. [Email Operations](#2-email-operations)
3. [CV/Resume Operations](#3-cvresume-operations)
4. [Letter/SOP Operations](#4-lettersop-operations)
5. [Alignment & Discovery](#5-alignment--discovery)
6. [Template Management](#6-template-management)
7. [Document Management](#7-document-management)
8. [Onboarding Flows](#8-onboarding-flows)
9. [Edge Cases & Error Handling](#9-edge-cases--error-handling)

---

## 1. Thread Lifecycle

### 1.1 Create New Thread

**Trigger**: User clicks "New Chat" or opens chat for a funding request

**Flow**:
```
POST /threads {funding_request_id}
  │
  ├─ Validate student owns request
  ├─ Create chat_thread record
  ├─ Load initial context
  ├─ Generate initial suggestions
  │
  └─ Return {thread_id, status: "active"}
```

**Initial Suggestions** (context-aware):

| Context | Suggestions |
|---------|-------------|
| No email draft | ["Draft an email to Professor X", "Check my alignment", "Help me improve my CV"] |
| Has email draft | ["Review my email", "Improve the introduction", "Check alignment"] |
| Email sent, no reply | ["Draft a reminder", "What should I do next?", "Review my application"] |
| Professor replied | ["Help me respond", "Analyze the reply", "What does this mean?"] |

### 1.2 Open Existing Thread

**Trigger**: User clicks thread in sidebar

**Flow**:
```
GET /threads/{id}/history
  │
  ├─ Load messages (paginated)
  ├─ Load thread metadata
  ├─ Check for summary (if long)
  ├─ Refresh context if stale (>1 hour)
  │
  └─ Return {messages, has_more, suggestions}
```

**Edge Cases**:
- Thread archived → Show read-only with "Reopen" option
- Thread failed → Show error with "Retry" option
- Thread running → Show progress indicator

### 1.3 Send Message

**Trigger**: User types and sends message

**Flow**:
```
POST /threads/{id}/messages {content}
  │
  ├─ Validate content (length, moderation)
  ├─ Save user message
  ├─ Route request (DIRECT/GUIDED/AGENTIC)
  ├─ Process with appropriate mode
  ├─ Stream response via SSE
  ├─ Save assistant message
  ├─ Generate new suggestions (async)
  │
  └─ SSE events until RESPONSE_END
```

**Rate Limiting**:
- Max 1 message per second per thread
- Max 100 messages per hour per student

---

## 2. Email Operations

### 2.1 Generate Email from Scratch

**User Input Examples**:
- "Write an email to this professor"
- "Draft an outreach email"
- "Help me contact Professor Smith"

**Flow**:
```
Route: DIRECT → email_generate
  │
  ├─ Build sender_detail from user context
  │   ├─ Name, email, phone
  │   ├─ Research interests
  │   ├─ Degrees and institutions
  │   ├─ Publications (if any)
  │   └─ Skills
  │
  ├─ Build recipient_detail from professor context
  │   ├─ Name, title, department
  │   ├─ Research areas
  │   ├─ Institution
  │   └─ Recent papers (if available)
  │
  ├─ Apply memory preferences
  │   ├─ Tone preference
  │   ├─ Topics to avoid
  │   └─ Topics to emphasize
  │
  ├─ Generate via EmailEngine.generate()
  │
  └─ Return formatted email with review offer
```

**Output Format**:
```markdown
I've drafted an email for Professor Smith:

**Subject:** Research Opportunity Inquiry - ML for Healthcare

Dear Professor Smith,

[body text]

Best regards,
[Student Name]

---
Would you like me to:
- Review this email for improvements
- Make specific changes
- Apply it to your request
```

**Edge Cases**:

| Scenario | Handling |
|----------|----------|
| No professor info | Error: "I need professor information first" |
| No user profile | Trigger data onboarding |
| Missing research connection | Generate generic intro, suggest adding connection |
| Professor on sabbatical (if known) | Warn user, suggest waiting |

### 2.2 Review Existing Email

**User Input Examples**:
- "Review my email"
- "Is this email ready to send?"
- "How can I improve this?"

**Flow**:
```
Route: DIRECT → email_review
  │
  ├─ Get email (from context or last generated)
  ├─ Load sender/recipient context
  ├─ Call EmailEngine.review()
  │   ├─ Compute 7-dimension scores
  │   └─ Determine readiness level
  │
  └─ Return structured feedback
```

**Output Format**:
```markdown
**Email Review Complete**

**Overall Score:** 7.5/10 (strong)

**Dimension Scores:**
• Subject Quality: 8/10
• Research Fit: 7/10
• Evidence Quality: 6/10
• Tone Appropriateness: 9/10
• Length Efficiency: 7/10
• Call to Action: 8/10
• Overall Strength: 8/10

**Key Strengths:**
- Clear subject line
- Professional tone
- Good closing

**Suggestions:**
1. Add specific paper reference
2. Strengthen research connection
3. Quantify achievements

Would you like me to help improve any of these areas?
```

**Readiness Levels**:
- **excellent (8.5-10)**: "Ready to send!"
- **strong (7-8.49)**: "Almost there, minor tweaks recommended"
- **needs_minor_revision (5-6.99)**: "Good foundation, some improvements needed"
- **needs_major_revision (1-4.99)**: "Significant work needed"

### 2.3 Optimize/Improve Email

**User Input Examples**:
- "Improve the introduction"
- "Make it more formal"
- "Add mention of [specific paper]"
- "Remove the part about my gap year"

**Flow**:
```
Route: AGENTIC (usually needs reasoning)
  │
  ├─ Identify what to change
  ├─ Get current email
  ├─ Call EmailEngine.generate() with optimization_context
  │   ├─ old_email
  │   ├─ feedback
  │   └─ revision_goals
  │
  └─ Return improved email with diff highlights
```

**Optimization Types**:

| Request | Optimization Type |
|---------|------------------|
| "More formal" | Tone adjustment |
| "Shorter" | Length optimization |
| "Add paper reference" | Content addition |
| "Remove X" | Content removal |
| "Stronger intro" | Section improvement |

### 2.4 Fill Request Fields

**User Input Examples**:
- "My research interest is machine learning for healthcare"
- "I want to connect through their Nature paper"
- "The paper title is 'Deep Learning for Drug Discovery'"

**Flow**:
```
Route: DIRECT or AGENTIC
  │
  ├─ Identify field being updated
  │   ├─ research_interest
  │   ├─ paper_title
  │   ├─ journal
  │   ├─ year
  │   └─ research_connection
  │
  ├─ Validate/enhance input
  ├─ Update funding_request record
  │
  └─ Confirm update + suggest next step
```

**Field Enhancement**:
```python
# AI can enhance vague inputs
"ML stuff" → "Machine Learning with focus on deep neural networks"
"that cancer paper" → Lookup and fill exact title/journal/year
```

### 2.5 Apply Email to Request

**User Input Examples**:
- "Apply this email"
- "Save this to my request"
- "Use this email"

**Flow**:
```
Route: DIRECT → email_apply
  │
  ├─ Get latest generated email
  ├─ Format for storage
  ├─ Update funding_emails table
  │   ├─ main_email_subject
  │   └─ main_email_body
  │
  ├─ Dispatch webhook: email.applied
  │
  └─ Confirm + offer to send
```

---

## 3. CV/Resume Operations

### 3.1 Generate CV

**User Input Examples**:
- "Create a CV for me"
- "Generate an academic resume"
- "Make a CV for this application"

**Flow**:
```
Route: DIRECT → resume_generate
  │
  ├─ Build user_details from context + memory
  │   ├─ Personal info
  │   ├─ Education history
  │   ├─ Work experience
  │   ├─ Publications
  │   ├─ Skills
  │   └─ Awards/achievements
  │
  ├─ Determine tone (academic/industry/clinical)
  ├─ Generate via CVEngine.generate()
  │
  └─ Return JSON + offer PDF/review
```

**Output Format**:
```markdown
I've generated your academic CV. Here's a summary:

**Personal Info:** [Name], [Current Institution]

**Education:**
- PhD in Computer Science, MIT (2024)
- MS in Data Science, Stanford (2020)

**Experience:** 3 positions
**Publications:** 5 papers
**Skills:** 12 technical skills

Would you like me to:
- Generate a PDF version
- Review it for improvements
- Customize it for this professor
```

### 3.2 Review CV

**Flow**:
```
Route: DIRECT → resume_review
  │
  ├─ Get CV (from context or last generated)
  ├─ Optionally get target context (professor/position)
  ├─ Call CVEngine.review()
  │   ├─ 7-dimension scoring
  │   └─ Section-specific feedback
  │
  └─ Return structured review
```

**Review Dimensions**:
1. Content Completeness
2. Research Presentation
3. Technical Depth
4. Publication Quality
5. Structure/Clarity
6. Target Alignment
7. Overall Strength

### 3.3 Optimize CV Section

**User Input Examples**:
- "Improve my education section"
- "Make my publications stand out"
- "Highlight my ML experience"

**Flow**:
```
Route: GUIDED → [resume_review, resume_optimize]
  │
  ├─ Identify sections to modify
  ├─ Get current CV
  ├─ Get feedback (implicit or explicit)
  ├─ Call CVEngine.optimize()
  │
  └─ Return optimized CV with changes highlighted
```

### 3.4 Export CV to PDF

**User Input Examples**:
- "Give me a PDF"
- "Download my CV"
- "Export as PDF"

**Flow**:
```
Route: DIRECT → resume_export
  │
  ├─ Get latest CV
  ├─ Render to LaTeX
  ├─ Compile PDF
  │   ├─ Try latexmk
  │   ├─ Fallback to xelatex
  │   └─ Fallback to pdflatex
  │
  ├─ Upload to S3 (sandbox)
  │
  └─ Return download link
```

**Compilation Error Handling**:
```markdown
I encountered an issue compiling your CV to PDF.

**Error:** Missing font "Times New Roman"

**Options:**
1. I can regenerate with a different font
2. You can download the LaTeX source and compile locally
3. I can try a simpler template

Which would you prefer?
```

---

## 4. Letter/SOP Operations

### 4.1 Generate SOP

**User Input Examples**:
- "Write my statement of purpose"
- "Create a motivation letter"
- "Help me with my SOP"

**Flow**:
```
Route: DIRECT → letter_generate
  │
  ├─ Build context
  │   ├─ User background
  │   ├─ Target program/professor
  │   ├─ Research goals
  │   └─ Unique strengths
  │
  ├─ Determine letter type (SOP/motivation/cover)
  ├─ Generate via LetterEngine.generate()
  │
  └─ Return structured letter
```

### 4.2 Review SOP

Similar to email review with different dimensions:
1. Narrative Flow
2. Research Vision
3. Personal Story
4. Program Fit
5. Writing Quality
6. Authenticity
7. Overall Impact

---

## 5. Alignment & Discovery

### 5.1 Evaluate Alignment

**User Input Examples**:
- "How well do I match with this professor?"
- "Check my alignment"
- "Am I a good fit?"

**Flow**:
```
Route: DIRECT → alignment_evaluate
  │
  ├─ Load user profile
  ├─ Load professor profile
  ├─ Call AlignmentEngine.evaluate()
  │   ├─ Research topic overlap
  │   ├─ Methodological alignment
  │   ├─ Skills compatibility
  │   └─ Publication match
  │
  └─ Return score + recommendations
```

**Output Format**:
```markdown
**Alignment Evaluation**

**Overall Score:** 7.2/10 (STRONG alignment)

**Category Breakdown:**
• Research Topics: 8.5/10
  - Both work on NLP, specifically transformers
• Methods: 6.5/10
  - You use PyTorch, professor's lab uses JAX
• Skills: 7.0/10
  - Strong overlap in deep learning
• Publications: 7.0/10
  - Both published in NeurIPS/ICML venues

**Strengths to Emphasize:**
1. Shared interest in transformer efficiency
2. Your paper on attention mechanisms aligns with their recent work
3. Experience with large-scale training

**Areas to Address:**
1. JAX experience (mention willingness to learn)
2. No direct healthcare application experience

**Recommendation:** Strong match! I recommend proceeding with outreach.
```

### 5.2 Compare Professors

**User Input Examples**:
- "Compare Professor Smith and Professor Jones"
- "Which professor is a better fit?"
- "Should I contact both?"

**⚠️ GAP: Not implemented**

**Required Flow**:
```
Route: AGENTIC
  │
  ├─ Get both professor contexts
  ├─ Evaluate alignment with each
  ├─ Compare scores dimension by dimension
  ├─ Consider external factors
  │   ├─ Lab size
  │   ├─ Funding status
  │   ├─ Location preferences
  │   └─ Response likelihood
  │
  └─ Return comparison + recommendation
```

**Output Format**:
```markdown
**Professor Comparison**

|  | Prof. Smith | Prof. Jones |
|--|------------|-------------|
| Alignment Score | 7.2/10 | 6.8/10 |
| Research Match | Strong | Moderate |
| Lab Size | 8 students | 3 students |
| Location | Boston | Toronto |

**Analysis:**
- Prof. Smith: Higher alignment, but larger lab (more competition)
- Prof. Jones: Good fit, smaller lab (more mentorship)

**Recommendation:** 
Contact both, but tailor your approach:
- Smith: Emphasize specific shared interests
- Jones: Emphasize desire for close mentorship
```

### 5.3 Find Similar Professors

**User Input Examples**:
- "Find more professors like this one"
- "Who else works on NLP in Canada?"
- "Recommend professors for me"

**Flow**:
```
Route: DIRECT → programs_recommend
  │
  ├─ Load user profile
  ├─ Apply filters (country, area, etc.)
  ├─ Query professors database
  ├─ Compute alignment scores
  ├─ Rank and return top N
  │
  └─ Return recommendations
```

---

## 6. Template Management

### 6.1 View Templates

**User Input Examples**:
- "Show me my email templates"
- "What templates do I have?"

**Flow**:
```
Route: DIRECT → template_list
  │
  ├─ Load user's templates from funding_student_templates
  ├─ Categorize (standard, detailed, reminder1-3)
  │
  └─ Return formatted list
```

### 6.2 Edit Template

**User Input Examples**:
- "Edit my standard template"
- "Update the subject line in my template"

**Flow**:
```
Route: AGENTIC
  │
  ├─ Identify which template
  ├─ Get current content
  ├─ Apply user's changes
  ├─ Validate (required variables, length)
  ├─ Update database
  │
  └─ Confirm changes
```

### 6.3 AI Improve Template

**User Input Examples**:
- "Make my template better"
- "Improve the opening"

**Flow**:
```
Route: DIRECT → template_improve
  │
  ├─ Load template
  ├─ Call TemplateAgent.suggest_improvements()
  ├─ Return suggestions + enhanced version
  │
  └─ Offer to apply changes
```

---

## 7. Document Management

### 7.1 Upload Document

**User Input Examples**:
- "Here's my resume" (with file attachment)
- "I'm uploading my CV"

**Flow**:
```
POST /documents (multipart form)
  │
  ├─ Validate file (type, size)
  ├─ Check for duplicates (hash)
  ├─ Upload to S3
  ├─ Create document record
  ├─ Queue processing job
  │
  └─ Return document_id + status
```

**Processing Pipeline**:
```
upload → extract_text → convert_to_json → validate → store
```

### 7.2 Apply Document to Request

**User Input Examples**:
- "Attach my CV to this request"
- "Apply this document"

**Flow**:
```
Route: DIRECT → document_apply
  │
  ├─ Get document
  ├─ Verify ownership
  ├─ Create attachment record
  ├─ Dispatch webhook
  │
  └─ Confirm attachment
```

### 7.3 Download Document

**Flow**:
```
GET /documents/{id}/download?format=pdf
  │
  ├─ Get document
  ├─ Check for exported PDF
  │   ├─ If exists: stream from S3
  │   └─ If not: compile and stream
  │
  └─ Return file stream
```

---

## 8. Onboarding Flows

### 8.1 Gmail Connection

**Flow**:
```
User: "Connect my Gmail"
  │
  ├─ Check current status
  ├─ If not started:
  │   └─ Return setup instructions
  │
  ├─ If has client_secret:
  │   └─ Return OAuth URL
  │
  ├─ If authorizing:
  │   └─ Wait for callback
  │
  └─ If complete:
      └─ Confirm connection
```

**Status Messages**:
- `not_started`: "To connect Gmail, you'll need to create OAuth credentials..."
- `client_secret`: "Upload your client_secret.json file..."
- `auth_required`: "Click this link to authorize access..."
- `complete`: "Gmail is connected!"

### 8.2 Profile Data Collection

**Flow**:
```
Missing profile data detected
  │
  ├─ Identify missing fields
  │   ├─ Basic info (name, email)
  │   ├─ Education
  │   ├─ Research interests
  │   └─ Skills
  │
  ├─ Ask for each missing field
  ├─ Validate responses
  ├─ Store to profile
  │
  └─ Complete when all required filled
```

**Conversation Example**:
```
Dana: I notice your profile is incomplete. Let me help you fill it out.
      What is your current educational level?

User: I'm a PhD student

Dana: Great! What field is your PhD in?

User: Computer Science, focusing on machine learning

Dana: Perfect. And which university are you at?
...
```

---

## 9. Edge Cases & Error Handling

### 9.1 User Cancellation

**Scenario**: User says "never mind" or "cancel" mid-task

**Handling**:
```python
if detect_cancellation(message):
    # Stop any running jobs
    await job_service.cancel_thread_jobs(thread_id)
    
    # Acknowledge
    return "No problem! Is there something else I can help with?"
```

### 9.2 Conflicting Instructions

**Scenario**: "Always be formal" followed by "Use a friendly tone"

**Handling**:
- Most recent instruction wins
- Warn user about conflict
- Update memory with new preference

```markdown
I notice you previously asked for formal emails. 
Should I update your preference to friendly, or would you like 
me to use friendly just this once?
```

### 9.3 Missing Context

**Scenario**: User asks about email but no request selected

**Handling**:
```markdown
I don't see a funding request selected. 

Would you like me to:
1. Help you create a new request?
2. Show your existing requests?
```

### 9.4 Long Response Needed

**Scenario**: Generated email or CV is very long

**Handling**:
- Break into sections
- Offer full view vs summary
- Enable progressive disclosure

```markdown
I've generated a comprehensive CV. Here's a summary:

**Sections:** 8 (Education, Experience, Publications, ...)
**Total items:** 47
**Est. pages:** 3

Would you like me to:
- Show the full CV
- Show specific sections
- Generate a 1-page version
```

### 9.5 API/Tool Failure

**Scenario**: LLM timeout or tool error

**Handling**:
```markdown
I encountered a temporary issue while generating your email.

[Error: Timeout after 60 seconds]

Options:
1. Try again (recommended)
2. Use a simpler request
3. Contact support

Would you like me to retry?
```

### 9.6 Ambiguous Request

**Scenario**: "Make it better" (what is "it"?)

**Handling**:
```markdown
I'd be happy to help improve something! 

Are you referring to:
1. The email we just drafted?
2. Your CV?
3. Something else?
```

### 9.7 Off-Topic Request

**Scenario**: "What's the weather?" or "Tell me a joke"

**Handling**:
```markdown
I specialize in helping with academic outreach and applications. 
I can help you with:
- Drafting emails to professors
- Creating and improving your CV
- Writing statements of purpose
- Evaluating research alignment

What would you like help with?
```

---

## Scenario Priority Matrix

| Scenario | Frequency | Complexity | Priority |
|----------|-----------|------------|----------|
| Generate email | Very High | Medium | P0 |
| Review email | Very High | Low | P0 |
| Generate CV | High | High | P0 |
| Check alignment | High | Medium | P0 |
| Edit email | High | Medium | P1 |
| Apply document | Medium | Low | P1 |
| Compare professors | Medium | High | P2 |
| Generate reminder | Medium | Medium | P1 |
| Template editing | Low | Medium | P2 |
| Document upload | Low | High | P1 |

---

## Implementation Checklist

- [x] Email generation
- [x] Email review
- [x] CV generation
- [x] CV review
- [x] Alignment evaluation
- [ ] Professor comparison
- [ ] Reminder generation
- [ ] Template editing via chat
- [ ] Document bundling
- [ ] Multi-language support


