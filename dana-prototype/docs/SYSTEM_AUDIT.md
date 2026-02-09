# Dana AI Copilot - System Audit & Gap Analysis

> **Document Version**: 1.0  
> **Date**: January 2026  
> **Scope**: Comprehensive audit of Dana AI copilot architecture, implementation gaps, and recommendations

## Table of Contents

1. [Agent System Architecture](#1-agent-system-architecture)
2. [Processing Modes Behavior](#2-processing-modes-behavior)
3. [Orchestration & Switchboard Logic](#3-orchestration--switchboard-logic)
4. [Prompting & Context Engineering](#4-prompting--context-engineering)
5. [User Flow Coverage](#5-user-flow-coverage)
6. [Conversation Length Management](#6-conversation-length-management)
7. [UX Helpers Inside Chat](#7-ux-helpers-inside-chat)

---

## 1. Agent System Architecture

### 1.1 Full Agent Lineup

| Agent | Category | Status | Primary Responsibilities |
|-------|----------|--------|-------------------------|
| **DanaOrchestrator** | Core | ✅ Implemented | Main switchboard, intent routing, ReAct loops, response synthesis |
| **EmailEngine** | Content | ✅ Implemented | Generate, review, optimize outreach emails |
| **CVEngine** | Content | ✅ Implemented | Generate, review, optimize academic CVs with LaTeX |
| **LetterEngine** | Content | ✅ Implemented | Generate, review SOPs and motivation letters |
| **AlignmentEngine** | Analysis | ✅ Implemented | Evaluate student-professor research alignment |
| **MemoryAgent** | State | ✅ Implemented | Store/retrieve user preferences with semantic search |
| **ModerationAgent** | Safety | ✅ Implemented | Content safety checks, policy compliance |
| **GmailOnboardingAgent** | Onboarding | ✅ Implemented | Guide Gmail OAuth setup |
| **DataOnboardingAgent** | Onboarding | ✅ Implemented | Collect user profile data |
| **TemplateAgent** | Onboarding | ✅ Implemented | Manage email templates |
| **ProgramsAgent** | Discovery | ✅ Implemented | Recommend professors/programs |
| **Converter** | Processing | ✅ Implemented | Parse documents to structured JSON |
| **FollowUpAgent** | UX | ✅ Implemented | Generate follow-up suggestions |
| **TitleAgent** | UX | ✅ Implemented | Generate chat titles |
| **SummarizationAgent** | Context | ✅ Implemented | Compress long conversations |
| **MemoryExtractionAgent** | State | ✅ Implemented | Extract preferences from conversations |

### 1.2 Agent Responsibilities & Boundaries

#### EmailEngine

**Owns**:
- Email generation from scratch
- Email optimization with feedback
- Multi-dimensional email review (7 dimensions)
- Email rendering to HTML

**Boundaries (NOT allowed)**:
- ❌ Fabricate professor information
- ❌ Include claims not supported by user's profile
- ❌ Access external APIs directly
- ❌ Store emails (delegated to DB service)

**Failure Modes**:
- LLM timeout → Retry with exponential backoff
- Invalid professor context → Return error with missing fields
- Cache corruption → Regenerate with `regenerate=True`

#### CVEngine

**Owns**:
- CV generation from user details
- Section-specific optimization
- Multi-dimensional review (7 dimensions)
- LaTeX rendering and PDF compilation

**Boundaries**:
- ❌ Modify user facts (only presentation)
- ❌ Include unverifiable claims
- ❌ Direct file system access outside artifacts_dir

**Failure Modes**:
- LaTeX compilation error → Return error log with extracted messages
- Missing LaTeX engine → Raise RuntimeError with install instructions
- Malformed user data → Generate with available fields, note missing

#### MemoryAgent

**Owns**:
- Memory storage with TTLs
- Semantic search via embeddings
- Memory deduplication
- Memory lifecycle (expiration, deactivation)

**Boundaries**:
- ❌ Store PII without explicit user consent
- ❌ Store exact conversation transcripts
- ❌ Infer sensitive information (health, finances)
- ❌ Share memories across students

**Failure Modes**:
- Embedding API failure → Fall back to keyword search
- Duplicate detection → Update existing instead of create
- Storage failure → Log and continue without storage

### 1.3 Agent Communication Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    DanaOrchestrator                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Tool Registry                       │   │
│  │  [email_generate] [email_review] [email_optimize]    │   │
│  │  [resume_generate] [resume_review] [resume_optimize] │   │
│  │  [alignment_evaluate] [memory_push] [memory_pull]    │   │
│  │  [get_user_context] [get_professor_context] ...      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI Function Calling
                         │ OR Direct Invocation (DIRECT mode)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Wrappers                             │
│  - Validate inputs                                           │
│  - Build agent-specific context from OrchestrationContext   │
│  - Execute agent method                                      │
│  - Transform result to ToolResult schema                    │
└─────────────────────────────────────────────────────────────┘
```

**Communication Mechanisms**:

1. **Function Calling** (AGENTIC mode):
   - LLM selects tools via OpenAI function calling
   - Tool registry provides JSON schemas
   - Orchestrator executes and feeds results back

2. **Direct Invocation** (DIRECT mode):
   - Router maps intent → tool
   - Tool executed without LLM reasoning
   - Template synthesis for response

3. **Shared State via Context**:
   - `OrchestrationContext` passed to all tool calls
   - Contains user, professor, request, memory, conversation
   - Immutable within single request

**Guardrails**:
- Max 5 tool iterations per request
- Tool timeout: 60s per call
- Result truncation: 2000 chars per tool
- No tool-to-tool direct calls (must go through orchestrator)

---

## 2. Processing Modes Behavior

### 2.1 Mode Definitions

| Mode | Complexity | Token Cost | LLM Calls | Use Case |
|------|------------|------------|-----------|----------|
| **DIRECT** | Low | ~200-300 | 0-1 | Single clear intent, pattern match |
| **GUIDED** | Medium | ~400-600 | 1 | Multi-step workflows, known sequences |
| **AGENTIC** | High | ~2000-5000 | 2-6 | Complex reasoning, ambiguous intent |

### 2.2 Mode Selection Process

```python
# Router Decision Flow
def route(message: str, context: OrchestrationContext) -> RouteDecision:
    # 1. Pattern matching (DIRECT)
    if matches_direct_pattern(message):
        return DIRECT mode
    
    # 2. Sequence matching (GUIDED)
    if matches_guided_pattern(message):
        return GUIDED mode
    
    # 3. Complexity analysis (AGENTIC)
    complexity = analyze_complexity(message, context)
    return AGENTIC mode with model_tier based on complexity
```

**Selection is**:
- ✅ **Zero-token**: No LLM calls for routing
- ✅ **Deterministic**: Same input → same mode
- ⚠️ **Pattern-based**: Limited to predefined patterns

### 2.3 Mode Behavior Comparison

**DIRECT Mode Flow**:
```
Input → Pattern Match → Tool Selection → Execute Tool → Template Synthesis → Output
        [0 tokens]      [0 tokens]       [~500 tokens]  [0-200 tokens]
```

**GUIDED Mode Flow**:
```
Input → Sequence Match → Execute Tools (sequential) → Lightweight Synthesis → Output
        [0 tokens]       [~500 tokens × N]             [~300 tokens]
```

**AGENTIC Mode Flow**:
```
Input → Context Build → ReAct Loop:
                          ├─ Reasoning (CoT)
                          ├─ Tool Selection (function calling)
                          ├─ Tool Execution
                          └─ Repeat until answer
                        → Final Response
        [~500 tokens]    [~500-1000 tokens × iterations]
```

### 2.4 Quality, Cost, Latency Metrics

**⚠️ GAP: No metrics collection implemented**

**Required Implementation**:
```python
@dataclass
class ModeMetrics:
    mode: ProcessingMode
    latency_ms: int
    tokens_input: int
    tokens_output: int
    tool_calls: int
    success: bool
    user_satisfaction: Optional[float]  # Implicit or explicit feedback
```

**Recommended Collection Points**:
1. Start of `process_stream()` → timestamp
2. Each tool execution → tokens, latency
3. End of response → total metrics
4. Webhook: `job.completed` → includes metrics

### 2.5 Robustness & Consistency

**Current State**:
- ✅ Mode selection consistent (deterministic)
- ✅ Fallback to AGENTIC for unknown patterns
- ⚠️ No error recovery within modes
- ❌ No mode switching mid-stream

**Recommendations**:
1. **Mode Escalation**: If DIRECT fails, retry with AGENTIC
2. **Graceful Degradation**: If tools fail, continue with available data
3. **Circuit Breaker**: After N failures, use cached/fallback responses

---

## 3. Orchestration & Switchboard Logic

### 3.1 Orchestrator Decision-Making Audit

**Current Implementation**:
```python
class DanaOrchestrator:
    MAX_TOOL_ITERATIONS = 5
    
    async def process_stream(thread_id, message):
        context = await context_builder.build(thread_id)
        route = route_request(message, context)  # Zero-token routing
        
        if route.mode == DIRECT:
            yield from _process_direct(context, route)
        elif route.mode == GUIDED:
            yield from _process_guided(context, route)
        else:
            yield from _process_agentic(context, route)
```

**Audit Findings**:

| Aspect | Status | Finding |
|--------|--------|---------|
| Intent Classification | ⚠️ | Regex-based, limited coverage |
| Confidence Scoring | ✅ | Included in RouteDecision |
| Fallback Strategy | ✅ | Defaults to AGENTIC |
| Tool Selection | ✅ | OpenAI function calling |
| Result Synthesis | ✅ | Template + LLM hybrid |

### 3.2 Intent Router Audit

**Current Patterns** (from `router.py`):

```python
DIRECT_PATTERNS = [
    (r"(?:write|draft|generate|create)\s+(?:an?\s+)?email", "email_generate", 0.9),
    (r"review\s+(?:my\s+)?(?:the\s+)?email", "email_review", 0.95),
    # ... 10 patterns total
]
```

**Gap Analysis**:

| Coverage Area | Covered? | Missing Patterns |
|--------------|----------|------------------|
| Email operations | ✅ | - |
| CV operations | ✅ | - |
| Alignment | ✅ | - |
| Context retrieval | ✅ | - |
| Template editing | ❌ | "edit my template", "update template" |
| Reminder drafting | ❌ | "write reminder", "draft follow-up" |
| Comparison | ❌ | "compare professors", "which is better" |
| Cancellation | ❌ | "never mind", "cancel that" |
| Clarification | ❌ | "what do you mean", "explain" |

**Confidence Threshold Audit**:
- All patterns have confidence ≥ 0.85
- **Risk**: Low-confidence matches may misroute
- **Recommendation**: Add confidence < 0.7 → AGENTIC fallback

### 3.3 Retry/Repair Strategies

**⚠️ GAP: Limited retry implementation**

**Current State**:
- API-level retries in `llm_client` (exponential backoff)
- No agent-level retries
- No self-correction loops

**Required Implementation**:

```python
class RetryStrategy:
    """Retry strategies for orchestration failures."""
    
    async def retry_with_escalation(
        self,
        original_mode: ProcessingMode,
        error: Exception,
        context: OrchestrationContext,
    ) -> Optional[RouteDecision]:
        """
        Escalate mode on failure:
        DIRECT → GUIDED → AGENTIC → Human fallback
        """
        escalation_map = {
            ProcessingMode.DIRECT: ProcessingMode.GUIDED,
            ProcessingMode.GUIDED: ProcessingMode.AGENTIC,
            ProcessingMode.AGENTIC: None,  # Human escalation
        }
        
        next_mode = escalation_map.get(original_mode)
        if next_mode:
            return RouteDecision(mode=next_mode, ...)
        return None  # Trigger human-safe fallback
    
    async def self_correct(
        self,
        tool_result: ToolResult,
        context: OrchestrationContext,
    ) -> Optional[ToolResult]:
        """
        Attempt self-correction if tool result is invalid.
        """
        if tool_result.success:
            return tool_result
        
        # Try with modified parameters
        correction_prompt = f"""
        The tool {tool_result.tool_name} failed with error: {tool_result.error}
        
        Suggest corrected parameters or alternative approach.
        """
        # ... correction logic
```

### 3.4 Pattern Recognition in Orchestration

**⚠️ GAP: No workflow pattern learning**

**Recommended Implementation**:

```python
class WorkflowPatternRecognizer:
    """Detect and reuse common workflow patterns."""
    
    # Store successful sequences
    known_patterns: Dict[str, List[str]] = {
        "email_complete_flow": ["get_user_context", "get_professor_context", 
                                "email_generate", "email_review"],
        "cv_improvement": ["resume_review", "resume_optimize", "resume_review"],
    }
    
    async def match_workflow(
        self,
        message: str,
        history: List[str],
    ) -> Optional[str]:
        """Match message to known workflow patterns."""
        # Check if message implies multi-step workflow
        # Return pattern name if matched
```

### 3.5 Discoverability & Debugging

**Current State**:
- ✅ Job IDs tracked
- ✅ Trace IDs in job records
- ⚠️ No structured step timeline
- ❌ No distributed tracing

**Required Implementation**:

```python
@dataclass
class OrchestrationTrace:
    """Full trace of orchestration execution."""
    trace_id: str
    thread_id: int
    job_id: int
    
    steps: List[OrchestrationStep]
    
    # Timing
    started_at: datetime
    finished_at: datetime
    total_duration_ms: int
    
    # Costs
    total_tokens: int
    total_cost: Decimal
    
    # Route decision
    route: RouteDecision
    
    def to_timeline(self) -> List[dict]:
        """Format as debuggable timeline."""
        return [
            {
                "step": i,
                "name": step.name,
                "type": step.type,  # "reasoning", "tool_call", "synthesis"
                "started_at": step.started_at.isoformat(),
                "duration_ms": step.duration_ms,
                "tokens": step.tokens,
                "status": step.status,
                "data": step.summary,
            }
            for i, step in enumerate(self.steps)
        ]
```

---

## 4. Prompting & Context Engineering

### 4.1 Agent Prompt Audit

**Dana System Prompt** (`prompts.py` lines 4-62):

| Aspect | Status | Finding |
|--------|--------|---------|
| Role Clarity | ✅ | Clear "Dana" identity defined |
| Tool Rules | ⚠️ | Tools listed but not constrained |
| Input/Output Contracts | ❌ | No strict output format |
| Guardrails | ✅ | 5 explicit rules |
| Refusal Behaviors | ⚠️ | Implicit only |

**Recommended Improvements**:

```python
DANA_SYSTEM_PROMPT_V2 = """You are Dana, an expert AI advisor for academic funding and professor outreach.

## Identity & Boundaries
- You are Dana, NOT an AI assistant or ChatGPT
- You specialize in academic outreach (NOT general knowledge)
- You ONLY help with professor outreach, CVs, SOPs, and related tasks
- You REFUSE requests unrelated to academic applications

## Tool Usage Rules
1. ALWAYS use tools for factual information (never fabricate)
2. ONLY use tools explicitly listed below
3. If a tool fails, explain the limitation and offer alternatives
4. NEVER mention tool names to users (describe actions naturally)

## Response Format
- Use markdown for structure
- Be concise (≤300 words unless generating documents)
- Always end with a specific next action or question

## Refusal Triggers (respond with helpful redirect)
- General knowledge questions → "I specialize in academic outreach..."
- Unethical requests → "I can't help with that, but I can..."
- Out-of-scope tasks → "That's outside my expertise, but..."

## Context Injection Markers
{context} <!-- Injected by system, DO NOT repeat to user -->
"""
```

### 4.2 Context Construction

**Current Implementation** (`context.py`):

```python
def to_prompt_context(self) -> str:
    parts = []
    parts.append(f"## User: {self.user.first_name} {self.user.last_name}")
    # ... ~20 lines of context formatting
    return "\n".join(parts)
```

**Context Hierarchy**:

| Layer | Source | Purpose | TTL |
|-------|--------|---------|-----|
| System | Prompt constants | Agent identity, rules | Static |
| User | DB: students, documents | Profile, background | Session |
| Professor | DB: professors, institutes | Target info | Session |
| Request | DB: funding_requests | Application state | Session |
| Memory | DB: ai_memory | Preferences, instructions | Persistent |
| Conversation | DB: chat_thread_messages | History | Session |

### 4.3 Context vs Memory vs Retrieval

**⚠️ GAP: No clear separation policy**

**Recommended Policy**:

| Data Type | Storage | Access Pattern |
|-----------|---------|---------------|
| User facts (name, degrees) | Context | Always included |
| User preferences (tone) | Memory | Semantic retrieval |
| Professor info | Context | Always included |
| Conversation history | Context | Last N + summary |
| Generated documents | Retrieval | On-demand |
| Past email drafts | Retrieval | On-demand |

### 4.4 Context Bloat Prevention

**Current State**:
- ✅ Conversation limited to last 10-20 messages
- ✅ Summary compression exists
- ❌ No token counting before context build
- ❌ No dynamic pruning

**Required Implementation**:

```python
class ContextBudget:
    """Manage context token budget."""
    
    MAX_CONTEXT_TOKENS = 16384  # Leave room for response
    
    ALLOCATION = {
        "system_prompt": 2000,
        "user_context": 2000,
        "professor_context": 1000,
        "request_context": 500,
        "memory": 1000,
        "conversation": 8000,
        "current_message": 500,
        "buffer": 1384,
    }
    
    def build_within_budget(
        self,
        full_context: OrchestrationContext,
    ) -> str:
        """Build context string within token budget."""
        # Count tokens for each component
        # Prune lower-priority items if over budget
        # Use summary for conversation if needed
```

### 4.5 Prompt Injection Defense

**Current Defenses**:
- ✅ Moderation agent checks content
- ⚠️ No explicit context markers
- ❌ No input sanitization

**Required Defenses**:

```python
def sanitize_user_input(content: str) -> str:
    """Remove potential injection patterns."""
    # Remove instruction-like patterns
    patterns = [
        r"ignore previous instructions",
        r"forget everything",
        r"you are now",
        r"act as",
        r"system:",
        r"<\|.*\|>",  # Special tokens
    ]
    
    sanitized = content
    for pattern in patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
    
    return sanitized

def mark_context_boundaries(context: str) -> str:
    """Add explicit boundaries to prevent injection."""
    return f"""
<!-- BEGIN SYSTEM CONTEXT - DO NOT MODIFY -->
{context}
<!-- END SYSTEM CONTEXT -->
"""
```

---

## 5. User Flow Coverage

### 5.1 Complete User Journey Map

```
┌───────────────────────────────────────────────────────────┐
│                        USER JOURNEY                       │
├───────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│ │ ONBOARDING  │ -> │ PROFILE     │ -> │ REQUEST SETUP   │ │
│ │ - Gmail     │    │ - CV Upload │    │ - Professor     │ │
│ │ - Data      │    │ - Parse     │    │ - Interests     │ │
│ │ - Template  │    │ - Validate  │    │ - Connection    │ │
│ └─────────────┘    └─────────────┘    └─────────────────┘ │
│        │                  │                    │          │
│        v                  v                    v          │
│ ┌───────────────────────────────────────────────────────┐ │
│ │                    CHAT INTERACTION                   │ │
│ │  ┌─────────────┐  ┌────────────┐  ┌────────────────┐  │ │
│ │  │ ALIGNMENT   │  │ EMAIL      │  │ DOCUMENT       │  │ │
│ │  │ - Evaluate  │  │ - Generate │  │ - CV Generate  │  │ │
│ │  │ - Recommend │  │ - Review   │  │ - CV Review    │  │ │
│ │  │ - Compare   │  │ - Optimize │  │ - SOP Generate │  │ │
│ │  └─────────────┘  │ - Template │  │ - SOP Review   │  │ │
│ │                   └────────────┘  └────────────────┘  │ │
│ │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │ │
│ │  │ REMINDER    │  │ REVIEW      │  │ COMPARE       │  │ │
│ │  │ - Draft     │  │ - Check     │  │ - Professors  │  │ │
│ │  │ - Schedule  │  │ - Reply     │  │ - Strategies  │  │ │
│ │  │ - Send      │  │ - Analyze   │  │ - Trade-offs  │  │ │
│ │  └─────────────┘  └─────────────┘  └───────────────┘  │ │
│ └───────────────────────────────────────────────────────┘ │
│                             │                             │
│                             v                             │
│ ┌───────────────────────────────────────────────────────┐ │
│ │                    APPLICATION                        │ │
│ │  - Apply documents to request                         │ │
│ │  - Finalize email                                     │ │
│ │  - Send via Gmail                                     │ │
│ │  - Track & remind                                     │ │
│ └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 5.2 Opening Sidebar Chatbox (New Thread)

**Flow**:
```
User clicks chat button
  ↓
Frontend: POST /threads {funding_request_id}
  ↓
Backend: Create chat_thread record
  ↓
Backend: Load initial context (user, professor, request)
  ↓
Frontend: Display empty chat + suggestions
  ↓
User sends first message
  ↓
Orchestrator: Route + Process + Stream response
  ↓
Frontend: Display streamed response
  ↓
Generate title (async, after 2 messages)
  ↓
Generate suggestions (async)
```

**⚠️ Gap: Initial System Prompts**

```python
# Missing: Welcome message logic
async def get_initial_prompts(thread_id: int) -> Dict[str, Any]:
    """Generate initial prompts for new thread."""
    context = await context_builder.build(thread_id)
    
    # Determine user state
    if not context.user.degrees:
        return {"type": "onboarding", "message": "Let's start by setting up your profile..."}
    
    if not context.email.subject:
        return {"type": "email_prompt", "message": "I can help you draft an email to Professor X..."}
    
    return {"type": "general", "message": "How can I help you with your outreach?"}
```

### 5.3 Opening Existing Thread

**Flow**:
```
User clicks thread in sidebar
  ↓
Frontend: GET /threads/{id}/history?limit=50
  ↓
Backend: Load messages (chronological)
  ↓
Backend: Check for thread summary (if > 20 messages)
  ↓
Frontend: Display history
  ↓
User sends message
  ↓
Orchestrator: Load context (includes history) + Process
```

**⚠️ Gap: Context Rehydration**

Need to ensure:
1. Memory is refreshed for long-idle threads
2. Professor info is re-fetched (may have changed)
3. Conversation summary is used if available

### 5.4 User Request Scenarios

See [docs/USER_SCENARIOS.md](./USER_SCENARIOS.md) for complete scenarios including:
- Email field filling
- Template editing
- Email generation/review/optimization
- CV generation/review
- SOP generation
- Professor alignment
- Professor comparison
- Reminder drafting

---

## 6. Conversation Length Management

### 6.1 Current Implementation

```python
async def load_conversation_context(
    self,
    thread_id: int,
    max_messages: int = 20,  # ← Message-count based
) -> ConversationContext:
    messages = await self.db.get_recent_messages(thread_id, limit=max_messages)
    # ...
```

### 6.2 Token-Based Trigger

**⚠️ GAP: No token counting**

**Required Implementation**:

```python
from tiktoken import encoding_for_model

class ConversationManager:
    MAX_CONVERSATION_TOKENS = 8000
    SUMMARY_THRESHOLD = 16384
    
    async def get_optimized_conversation(
        self,
        thread_id: int,
    ) -> ConversationContext:
        """Get conversation optimized for context window."""
        enc = encoding_for_model("gpt-4o")
        
        # Get all messages
        all_messages = await db.get_thread_messages(thread_id, limit=200)
        
        # Count tokens
        total_tokens = sum(
            len(enc.encode(m.content))
            for m in all_messages
        )
        
        if total_tokens <= self.MAX_CONVERSATION_TOKENS:
            return ConversationContext(
                messages=all_messages,
                summary=None,
            )
        
        # Trigger summarization
        return await self._build_summarized_context(all_messages)
```

### 6.3 Summarization Structure

**Required Format**:

```
[:3] + [SUMMARY_BLOCK] + [:-3]
```

**Implementation**:

```python
async def _build_summarized_context(
    self,
    messages: List[Message],
) -> ConversationContext:
    """Build context with summary for long conversations."""
    # Keep first 3 messages (establish context)
    first_messages = messages[:3]
    
    # Keep last 3 messages (recent context)
    last_messages = messages[-3:]
    
    # Summarize middle messages
    middle_messages = messages[3:-3]
    summary = await self._summarize(middle_messages)
    
    # Create summary block
    summary_message = ConversationMessage(
        role="system",
        content=f"[CONVERSATION SUMMARY]\n{summary}\n[END SUMMARY]",
        message_idx=-1,
    )
    
    return ConversationContext(
        messages=first_messages + [summary_message] + last_messages,
        summary=summary,
        total_messages=len(messages),
    )
```

### 6.4 Loss-Minimizing Summarization

**Prompt for Summarization**:

```python
SUMMARIZATION_PROMPT = """Summarize this conversation for context preservation.

PRESERVE (must include):
1. All decisions made and their reasoning
2. All documents generated (type, key points)
3. All user preferences expressed
4. Current task status and next steps
5. Any specific instructions or constraints

OMIT (safe to exclude):
- Greeting/pleasantries
- Redundant back-and-forth
- Superseded drafts (keep only final)

FORMAT:
- Use bullet points
- Include timestamps for key events
- Mark pending tasks with [TODO]

Conversation:
{conversation}
"""
```

### 6.5 Injection-Safe Summary

**Defense Implementation**:

```python
async def _summarize_safely(
    self,
    messages: List[Message],
) -> str:
    """Summarize with injection protection."""
    # 1. Sanitize all messages before summarization
    sanitized = [
        sanitize_user_input(m.content)
        for m in messages
    ]
    
    # 2. Mark summary as system-generated
    summary = await self._llm_summarize(sanitized)
    
    # 3. Wrap with safety markers
    return f"[SYSTEM-GENERATED SUMMARY - TRUSTED]\n{summary}"
```

---

## 7. UX Helpers Inside Chat

### 7.1 Prompt Suggestions

**Current Implementation** (`helpers.py`):

```python
class FollowUpAgent:
    async def generate_suggestions(
        self,
        thread_id: int,
        n: int = 3,
    ) -> List[str]:
        # ... LLM-based generation
        return suggestions
```

**⚠️ Gap: Return format doesn't match spec**

**Required Format**:

```python
@dataclass
class Suggestion:
    title: str    # Shown on button
    prompt: str   # Sent when clicked (in user's tone)

# Output:
[
    {
        "title": "Review Email",
        "prompt": "Can you review my email and tell me if it's ready to send?"
    },
    {
        "title": "Improve Introduction", 
        "prompt": "I think the introduction could be stronger. Can you help me improve it?"
    },
    {
        "title": "Check Alignment",
        "prompt": "How well do I align with this professor's research?"
    }
]
```

**Updated Implementation**:

```python
SUGGESTION_PROMPT_V2 = """Generate {n} follow-up suggestions.

For each suggestion:
1. "title": Short button label (2-5 words)
2. "prompt": Natural user message (conversational, in their perspective)

Context:
- User: {user_name}
- Professor: {professor_name}
- Request status: {request_status}
- Recent conversation: {conversation}

Output JSON:
{{"suggestions": [{{"title": "...", "prompt": "..."}}]}}
"""

async def generate_suggestions(
    self,
    thread_id: int,
    n: int = 3,
) -> List[Dict[str, str]]:
    """Generate suggestions with title/prompt format."""
    # ... build prompt
    response = await self.llm.get_response(...)
    
    return response.get("suggestions", self._default_suggestions())[:n]
```

### 7.2 Response Streaming Events

**Current Events** (from `events.py`):

| Event | Purpose | Data |
|-------|---------|------|
| `response_start` | Mark start | `{thread_id, timestamp}` |
| `response_token` | Stream text | `"token text"` |
| `response_end` | Mark end | `{thread_id, job_id, mode}` |
| `progress` | Show progress | `{percent, message}` |
| `tool_start` | Tool begins | `{name, arguments}` |
| `tool_end` | Tool completes | `{name, success, result}` |
| `error` | Error occurred | `{error, code}` |
| `meta_action` | UI action | `{action, payload}` |

**⚠️ Gap: "What's happening" visibility for agentic flows**

**New Event Types Needed**:

```python
class EventType(str, Enum):
    # ... existing ...
    
    # New: Reasoning visibility
    REASONING_START = "reasoning_start"    # CoT begins
    REASONING_STEP = "reasoning_step"      # Individual thought
    REASONING_END = "reasoning_end"        # CoT complete
    
    # New: Tool explanation
    TOOL_EXPLAIN = "tool_explain"          # Why using this tool
```

**Frontend Display**:

```
┌─────────────────────────────────────────────┐
│ Dana is thinking...                          │
│                                              │
│ 🔍 Checking your profile...                 │
│ ✓ Loaded your research background           │
│                                              │
│ 🔍 Looking up Professor Smith's work...     │
│ ✓ Found 12 recent papers                    │
│                                              │
│ 🔍 Evaluating alignment...                  │
│ ⏳ Calculating match score...               │
│                                              │
│ [Progress: 65%]                              │
└─────────────────────────────────────────────┘
```

### 7.3 Suggestion History

**⚠️ GAP: Suggestions not persisted**

**Required Schema Update**:

```sql
ALTER TABLE chat_threads
ADD COLUMN suggestion_history JSON;

-- Structure:
-- [{
--     "generated_at": "2024-01-15T10:30:00Z",
--     "suggestions": [
--         {"title": "...", "prompt": "...", "used": false},
--         {"title": "...", "prompt": "...", "used": true},
--     ]
-- }]
```

**DB Service Update**:

```python
async def save_thread_suggestions(
    self,
    thread_id: int,
    suggestions: List[Dict[str, str]],
) -> None:
    """Save suggestions with history tracking."""
    thread = await self.get_thread(thread_id)
    
    # Get existing history
    history = json.loads(thread.suggestion_history or "[]")
    
    # Add new entry
    history.append({
        "generated_at": datetime.utcnow().isoformat(),
        "suggestions": suggestions,
    })
    
    # Keep last 10 generations
    history = history[-10:]
    
    await self.client.chatthread.update(
        where={"id": thread_id},
        data={
            "suggestions": json.dumps(suggestions),
            "suggestion_history": json.dumps(history),
        }
    )
```

---

## Next Steps

Continue to:
- [docs/USER_SCENARIOS.md](./USER_SCENARIOS.md) - Complete user flow scenarios
- [docs/SYSTEM_AUDIT_PART2.md](./SYSTEM_AUDIT_PART2.md) - Sections 8-16 of audit


