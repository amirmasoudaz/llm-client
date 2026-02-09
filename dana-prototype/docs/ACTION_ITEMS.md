# Dana AI Copilot - Action Items & Implementation Plan

> Prioritized list of gaps to address, based on System Audit findings

## Priority Levels

- **P0 (Critical)**: Must fix before launch
- **P1 (High)**: Launch + 1 week
- **P2 (Medium)**: Launch + 1 month
- **P3 (Low)**: Future iteration

---

## P0 - Critical (Pre-Launch)

### 1. Token-Based Conversation Summarization
**Current**: Message-count based (20 messages)  
**Required**: Token-based (16,384 tokens)  
**File**: `src/agents/orchestrator/context.py`

```python
# Implementation needed:
- Add tiktoken dependency
- Count tokens before building context
- Implement [:3] + [SUMMARY] + [:-3] structure
- Ensure summary is injection-safe
```
**Effort**: 4 hours

---

### 2. Suggestion Format Update
**Current**: `List[str]`  
**Required**: `List[{title: str, prompt: str}]`  
**Files**: 
- `src/agents/orchestrator/helpers.py`
- `src/services/db.py`
- `prisma/schema.prisma`

```python
# Implementation needed:
- Update FollowUpAgent.generate_suggestions()
- Update prompt template
- Add suggestion_history column to chat_threads
- Update API response schema
```
**Effort**: 3 hours

---

### 3. Credit Check Integration
**Current**: Mock implementation  
**Required**: Platform API integration  
**Files**:
- `src/services/usage.py`
- `src/services/db.py`

```python
# Implementation needed:
- Add platform API client
- Implement check_and_reserve()
- Implement finalize_job_cost()
- Add credit display to SSE events
```
**Effort**: 6 hours

---

### 4. Tool Failure Recovery
**Current**: Exception propagates  
**Required**: Retry with fallback  
**File**: `src/agents/orchestrator/engine.py`

```python
# Implementation needed:
- Add execute_tool_with_recovery()
- Implement mode escalation (DIRECT → GUIDED → AGENTIC)
- Add graceful degradation messages
```
**Effort**: 4 hours

---

### 5. Input Sanitization
**Current**: Basic moderation  
**Required**: Full injection defense  
**Files**:
- `src/agents/moderation/engine.py`
- `src/agents/orchestrator/engine.py`

```python
# Implementation needed:
- Add sanitize_user_input() function
- Add context boundary markers
- Filter injection patterns before processing
```
**Effort**: 3 hours

---

## P1 - High Priority (Launch + 1 Week)

### 6. Intent Router Expansion
**Current**: 10 patterns  
**Required**: ~30 patterns  
**File**: `src/agents/orchestrator/router.py`

**Missing Patterns**:
```python
DIRECT_PATTERNS += [
    # Template operations
    (r"(?:edit|update|change)\s+(?:my\s+)?template", "template_edit", 0.85),
    (r"show\s+(?:me\s+)?(?:my\s+)?templates?", "template_list", 0.90),
    
    # Reminder operations
    (r"(?:write|draft|send)\s+(?:a\s+)?reminder", "reminder_generate", 0.90),
    (r"should\s+i\s+(?:send\s+)?(?:a\s+)?(?:follow|reminder)", "reminder_check", 0.85),
    
    # Cancellation
    (r"never\s*mind|cancel|stop|forget\s*it", "cancel_action", 0.95),
    
    # Clarification
    (r"what\s+do\s+you\s+mean|explain|clarify", "clarify_request", 0.90),
    
    # Comparison
    (r"compare\s+(?:these\s+)?professors?", "professor_compare", 0.85),
]
```
**Effort**: 2 hours

---

### 7. Reminder Generation
**Current**: Not implemented  
**Required**: Full reminder workflow  
**New File**: `src/agents/email/reminders.py`

```python
# Implementation needed:
- ReminderEngine class
- should_send_reminder() check
- generate_reminder() with tone progression
- Integration with orchestrator tools
```
**Effort**: 8 hours

---

### 8. Reply Classification
**Current**: Not implemented  
**Required**: Reply analysis agent  
**New File**: `src/agents/email/reply_classifier.py`

```python
# Implementation needed:
- ReplyClassifier class
- classify_reply() with types
- sentiment analysis
- next_steps generation
```
**Effort**: 6 hours

---

### 9. Document Bundling
**Current**: Not implemented  
**Required**: Multi-doc PDF merge  
**New File**: `src/services/bundler.py`

```python
# Implementation needed:
- DocumentBundler class
- PDF merging with PyPDF2
- Cover page generation
- S3 storage integration
```
**Effort**: 6 hours

---

### 10. Observability Setup
**Current**: Basic logging  
**Required**: Full metrics + structured logs  
**Files**:
- New: `src/observability/metrics.py`
- New: `src/observability/logging.py`

```python
# Implementation needed:
- Prometheus metrics endpoints
- Structured JSON logging
- Trace ID propagation
- Cost tracking metrics
```
**Effort**: 8 hours

---

## P2 - Medium Priority (Launch + 1 Month)

### 11. Qdrant Integration
**Current**: MySQL blob for embeddings  
**Required**: Qdrant vector DB  
**Files**:
- New: `src/services/vector.py`
- Update: `src/agents/memory/engine.py`

**Effort**: 12 hours

---

### 12. Professor Comparison
**Current**: Not implemented  
**Required**: Side-by-side comparison  
**Files**:
- New tool wrapper in `src/agents/orchestrator/tool_wrappers.py`
- Update `src/agents/programs/engine.py`

**Effort**: 6 hours

---

### 13. Multi-Language Support
**Current**: English only  
**Required**: Language detection + adaptation  
**New File**: `src/services/language.py`

**Effort**: 8 hours

---

### 14. Workflow Pattern Learning
**Current**: Static sequences  
**Required**: Dynamic pattern recognition  
**New File**: `src/agents/orchestrator/patterns.py`

**Effort**: 12 hours

---

### 15. Document Versioning
**Current**: No history  
**Required**: Full version history  
**Files**:
- Update: `prisma/schema.prisma` (add versions table)
- Update: `src/services/db.py`

**Effort**: 8 hours

---

### 16. Checkpoint & Resume
**Current**: Not implemented  
**Required**: Long job checkpointing  
**New File**: `src/services/checkpoint.py`

**Effort**: 6 hours

---

## P3 - Future (Post-Launch)

### 17. A/B Testing for Prompts
Track which prompts perform better based on user feedback.

### 18. User Feedback Collection
Implicit (edit frequency) and explicit (thumbs up/down) feedback.

### 19. Automatic Prompt Optimization
Use feedback to improve prompts over time.

### 20. Cross-Student Analytics
Aggregate insights for system improvement.

### 21. Professor Response Prediction
ML model to predict reply likelihood.

---

## Implementation Order

```
Week 0 (Pre-Launch):
├─ Day 1-2: #5 Input Sanitization, #4 Tool Failure Recovery
├─ Day 3-4: #1 Token-Based Summarization
├─ Day 5: #2 Suggestion Format Update
└─ Day 6-7: #3 Credit Check Integration

Week 1 (Post-Launch):
├─ Day 1-2: #6 Intent Router Expansion
├─ Day 3-5: #7 Reminder Generation
├─ Day 6-7: #8 Reply Classification

Week 2:
├─ Day 1-2: #9 Document Bundling
└─ Day 3-5: #10 Observability Setup

Week 3-4:
├─ #11 Qdrant Integration
├─ #12 Professor Comparison
└─ #13 Multi-Language Support

Week 5-6:
├─ #14 Workflow Pattern Learning
├─ #15 Document Versioning
└─ #16 Checkpoint & Resume
```

---

## Resource Estimates

| Priority | Items | Total Hours | FTE Days |
|----------|-------|-------------|----------|
| P0 | 5 | 20 | 2.5 |
| P1 | 5 | 30 | 3.75 |
| P2 | 6 | 52 | 6.5 |
| P3 | 5 | TBD | TBD |
| **Total (P0-P2)** | **16** | **102** | **~13** |

---

## Risk Assessment

| Item | Risk if Delayed | Mitigation |
|------|-----------------|------------|
| Token Summarization | Context overflow | Reduce max history temporarily |
| Credit Integration | Revenue leakage | Manual billing reconciliation |
| Input Sanitization | Security breach | Enhanced monitoring |
| Tool Recovery | Poor UX | Clear error messages |
| Observability | Blind spots | Console logging |

---

## Dependencies

```
#1 (Token Summarization) → no deps
#2 (Suggestions) → no deps
#3 (Credits) → Platform API ready
#4 (Tool Recovery) → no deps
#5 (Sanitization) → no deps
#6 (Router) → no deps
#7 (Reminders) → Email engine ready ✓
#8 (Reply Classifier) → no deps
#9 (Bundler) → PyPDF2 installed
#10 (Observability) → Prometheus/Grafana infra
#11 (Qdrant) → Qdrant deployment
```

---

## Testing Requirements

### P0 Items
- Unit tests for all new functions
- Integration tests for full flow
- Security tests for sanitization
- Load tests for summarization

### P1 Items
- Unit tests
- Integration tests
- E2E tests for new user flows

### P2 Items
- Unit tests
- Performance benchmarks
- A/B testing setup


