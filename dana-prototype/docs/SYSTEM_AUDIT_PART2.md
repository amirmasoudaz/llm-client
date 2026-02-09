# Dana AI Copilot - System Audit Part 2

> Continuation of [SYSTEM_AUDIT.md](./SYSTEM_AUDIT.md) covering sections 8-16

## Table of Contents

8. [Document Generation & Compilation](#8-document-generation--compilation)
9. [Document Upload Processing Pipeline](#9-document-upload-processing-pipeline)
10. [Memory & Personalization Policy](#10-memory--personalization-policy)
11. [Retrieval, Embeddings & Caching](#11-retrieval-embeddings--caching)
12. [Moderation & Malicious Behavior Defense](#12-moderation--malicious-behavior-defense)
13. [Credits, Usage Metering & Policy](#13-credits-usage-metering--policy)
14. [Email Workflow Intelligence](#14-email-workflow-intelligence)
15. [Scalability & Reliability](#15-scalability--reliability)
16. ["Anything Missing" Sweep](#16-anything-missing-sweep)

---

## 8. Document Generation & Compilation

### 8.1 PDF Generation Pipeline

**Current Implementation** (`resume/engine.py`):

```python
async def compile_pdf(
    self,
    cv: Dict[str, Any],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # 1. Render LaTeX
    tex_content, bib_content, cls_content = self.render_latex(cv)
    
    # 2. Write files
    await self._write_file(tex_path, tex_content)
    await self._write_file(bib_path, bib_content)
    await self._write_file(cls_path, cls_content)
    
    # 3. Compile
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name]
    # ...
```

**Supported Outputs**:
- ✅ PDF (via LaTeX compilation)
- ✅ LaTeX source files
- ❌ DOCX (not implemented)
- ❌ HTML export

### 8.2 Email Injection into Request

**⚠️ GAP: No direct email injection**

**Required Implementation**:

```python
class EmailApplicationService:
    """Apply generated emails to funding requests."""
    
    async def apply_email_to_request(
        self,
        request_id: int,
        email: Dict[str, Any],
        email_type: Literal["main", "reminder_1", "reminder_2", "reminder_3"],
    ) -> bool:
        """Inject generated email into request body."""
        
        # Get email fields
        subject = email.get("subject", "")
        body = self._format_email_body(email)
        
        # Update funding_emails table
        if email_type == "main":
            await self.db.client.fundingemail.update(
                where={"funding_request_id": request_id},
                data={
                    "main_email_subject": subject,
                    "main_email_body": body,
                }
            )
        elif email_type.startswith("reminder"):
            reminder_num = email_type[-1]
            await self.db.client.fundingemail.update(
                where={"funding_request_id": request_id},
                data={
                    f"reminder_{reminder_num}_subject": subject,
                    f"reminder_{reminder_num}_body": body,
                }
            )
        
        return True
    
    def _format_email_body(self, email: Dict[str, Any]) -> str:
        """Format email dict to body string."""
        parts = []
        
        if email.get("greeting"):
            parts.append(email["greeting"])
        
        parts.append(email.get("body", ""))
        
        if email.get("closing"):
            parts.append(email["closing"])
        
        if email.get("signature_name"):
            signature = [email["signature_name"]]
            if email.get("signature_email"):
                signature.append(email["signature_email"])
            parts.append("\n".join(signature))
        
        return "\n\n".join(parts)
```

### 8.3 Bundle Generation

**⚠️ GAP: No multi-document bundling**

**Required Implementation**:

```python
class DocumentBundler:
    """Bundle multiple documents into single PDF."""
    
    async def create_bundle(
        self,
        student_id: int,
        request_id: int,
        document_ids: List[int],
        include_cover: bool = True,
    ) -> Path:
        """
        Create PDF bundle with:
        - Optional cover page
        - CV
        - SOP/Cover letter
        - Transcripts
        - Portfolio samples
        """
        pdf_paths = []
        
        # Generate cover page if requested
        if include_cover:
            cover_path = await self._generate_cover_page(student_id, request_id)
            pdf_paths.append(cover_path)
        
        # Get document PDFs
        for doc_id in document_ids:
            doc = await self.db.get_document(doc_id)
            if doc.exported_pdf_path:
                pdf_content = await self.storage.get_final(doc.exported_pdf_path)
                temp_path = self._save_temp(pdf_content)
                pdf_paths.append(temp_path)
        
        # Merge PDFs
        merged_path = await self._merge_pdfs(pdf_paths, student_id, request_id)
        
        return merged_path
    
    async def _merge_pdfs(
        self,
        pdf_paths: List[Path],
        student_id: int,
        request_id: int,
    ) -> Path:
        """Merge multiple PDFs into one."""
        from PyPDF2 import PdfMerger
        
        merger = PdfMerger()
        for path in pdf_paths:
            merger.append(str(path))
        
        output_path = self.temp_dir / f"bundle_{student_id}_{request_id}.pdf"
        merger.write(str(output_path))
        merger.close()
        
        return output_path
```

---

## 9. Document Upload Processing Pipeline

### 9.1 Current Pipeline

**Converter** (`converter/engine.py`):

```python
class Converter:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tex"}
    
    async def convert(self) -> dict:
        # 1. Load and parse (cache by file hash)
        raw_content = await self._load_parsed(cache=True)
        
        # 2. Convert to structured JSON
        content_hash = blake3(raw_content.encode()).hexdigest()[:16]
        
        # 3. Check cache
        cached = await self._read_converter_cache(cache_path)
        if cached:
            return cached
        
        # 4. LLM extraction
        response = await self.converter_model.get_response(
            messages=messages,
            response_format=self.schema,
        )
        
        return response.get("output", response)
```

### 9.2 Deduplication Strategy

**Current State**:
- ✅ Source file hash stored (`source_file_hash`)
- ✅ Extracted text hash planned (`extracted_text_hash`)
- ✅ Processed content hash planned (`processed_content_hash`)
- ❌ Cross-matching not implemented

**Required Implementation**:

```python
class DocumentDeduplicationService:
    """Dedupe and reprocessing avoidance for documents."""
    
    async def check_duplicate(
        self,
        student_id: int,
        file_content: bytes,
        document_type: str,
    ) -> Optional[int]:
        """
        Check if document already exists.
        Returns existing document_id if duplicate found.
        """
        # 1. Hash the source file
        source_hash = blake3(file_content).hexdigest()
        
        # 2. Check for exact match
        existing = await self.db.client.studentdocument.find_first(
            where={
                "student_id": student_id,
                "source_file_hash": source_hash,
                "document_type": document_type,
            }
        )
        
        if existing:
            return existing.id
        
        return None
    
    async def check_content_duplicate(
        self,
        student_id: int,
        extracted_text: str,
        document_type: str,
    ) -> Optional[int]:
        """
        Check if extracted content matches existing document.
        Catches reformatted versions of same content.
        """
        text_hash = blake3(extracted_text.encode()).hexdigest()
        
        existing = await self.db.client.studentdocument.find_first(
            where={
                "student_id": student_id,
                "extracted_text_hash": text_hash,
                "document_type": document_type,
            }
        )
        
        return existing.id if existing else None
    
    async def should_reprocess(
        self,
        document_id: int,
        processor_version: str,
    ) -> bool:
        """
        Determine if document needs reprocessing.
        
        Triggers:
        - Processor version changed
        - Processing failed previously
        - User requested re-extraction
        """
        doc = await self.db.get_document(document_id)
        
        if not doc:
            return True
        
        # Version mismatch
        if doc.processor_version != processor_version:
            return True
        
        # Previous failure
        if doc.upload_status == "failed":
            return True
        
        # Missing processed content
        if not doc.processed_content:
            return True
        
        return False
```

### 9.3 Parsed Output Storage

**Storage Hierarchy**:

| Data | Storage | Purpose |
|------|---------|---------|
| Raw file | S3 `sources/` | Original upload |
| Extracted text | DB `raw_extracted_text` | Search/debug |
| Processed JSON | DB `processed_content` | Structured data |
| Embeddings | Qdrant (future) | Semantic search |
| PDF export | S3 `documents/` | Download |

**Cache Layers**:

| Layer | TTL | Purpose |
|-------|-----|---------|
| Parser cache (disk) | Permanent | Raw extraction |
| Converter cache (disk) | Permanent | JSON conversion |
| Redis cache | 24h | Hot data |
| DB cache (processed_content) | Permanent | Structured data |

---

## 10. Memory & Personalization Policy

### 10.1 When to Store Memory

**Trigger Conditions**:

| Trigger | Memory Type | Auto-Store | Confidence |
|---------|-------------|------------|------------|
| User says "always use formal tone" | `tone` | Yes | 0.95 |
| User says "don't mention my gap year" | `do_dont` | Yes | 0.90 |
| User corrects Dana's response | `instruction` | Yes | 0.85 |
| User states career goal | `goal` | Yes | 0.80 |
| User provides background fact | `bio` | Yes | 0.80 |
| Dana infers preference from patterns | Any | If conf > 0.7 | Variable |

**Store Triggers in Code**:

```python
MEMORY_TRIGGERS = {
    # Explicit statements
    r"always\s+(use|write|keep)": {"type": "instruction", "confidence": 0.95},
    r"never\s+(mention|include|use)": {"type": "do_dont", "confidence": 0.90},
    r"don't\s+(mention|include|use)": {"type": "do_dont", "confidence": 0.90},
    r"prefer\s+(formal|friendly|academic)": {"type": "tone", "confidence": 0.90},
    r"my\s+goal\s+is": {"type": "goal", "confidence": 0.85},
    r"i\s+(have|completed|worked)": {"type": "bio", "confidence": 0.75},
}
```

### 10.2 What to Store (and Never Store)

**Store**:
- ✅ Tone preferences ("formal", "friendly")
- ✅ Topics to avoid (gap years, specific experiences)
- ✅ Topics to emphasize (specific skills, publications)
- ✅ Career goals (PhD, industry, specific labs)
- ✅ Writing style preferences (length, structure)
- ✅ Communication preferences (response format)

**Never Store**:
- ❌ Full conversation transcripts
- ❌ PII (SSN, passport numbers)
- ❌ Financial information
- ❌ Health information
- ❌ Relationship status
- ❌ Political/religious views
- ❌ Exact quotes from private communications

**Content Filter**:

```python
FORBIDDEN_MEMORY_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{9}\b",              # Passport
    r"password|secret|pin",
    r"credit\s*card|bank\s*account",
    r"health|medical|diagnosis",
    r"(boy|girl)friend|spouse|partner",
    r"political|religious|voted",
]

def is_safe_to_store(content: str) -> bool:
    """Check if content is safe to store as memory."""
    content_lower = content.lower()
    
    for pattern in FORBIDDEN_MEMORY_PATTERNS:
        if re.search(pattern, content_lower):
            return False
    
    return True
```

### 10.3 Memory Context Integration

**Safe Integration**:

```python
def build_memory_context(
    memories: List[Memory],
    max_memories: int = 10,
    max_tokens: int = 1000,
) -> str:
    """Build memory context safely."""
    
    # Sort by confidence and recency
    sorted_memories = sorted(
        memories,
        key=lambda m: (m.confidence, m.created_at),
        reverse=True,
    )[:max_memories]
    
    # Group by type
    by_type = defaultdict(list)
    for m in sorted_memories:
        by_type[m.memory_type].append(m.content)
    
    # Format with clear markers
    parts = ["## User Preferences (from previous conversations)"]
    
    if by_type.get("instruction"):
        parts.append("\n### Instructions:")
        parts.extend(f"- {i}" for i in by_type["instruction"][:3])
    
    if by_type.get("tone"):
        parts.append("\n### Tone Preferences:")
        parts.extend(f"- {t}" for t in by_type["tone"][:2])
    
    if by_type.get("do_dont"):
        parts.append("\n### Topics to Avoid/Emphasize:")
        parts.extend(f"- {d}" for d in by_type["do_dont"][:5])
    
    # Add safety note
    parts.append("\n(Apply these preferences but do not reference them explicitly)")
    
    return "\n".join(parts)
```

---

## 11. Retrieval, Embeddings & Caching

### 11.1 Embedding Strategy

**Current State**:
- ✅ Memory embeddings (text-embedding-3-small)
- ❌ Document embeddings not implemented
- ❌ Qdrant not integrated

**Recommended Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                      Embedding Strategy                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Qdrant Collections:                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ memories_{student_id}                                │    │
│  │ - user preferences, instructions                     │    │
│  │ - small chunks (1 memory = 1 vector)                │    │
│  │ - TTL-aware (filter expired)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ documents_{student_id}                               │    │
│  │ - CV sections, SOP paragraphs                       │    │
│  │ - chunk_size=512 tokens, overlap=50                 │    │
│  │ - metadata: doc_id, section, created_at             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ professors_global                                    │    │
│  │ - professor research descriptions                   │    │
│  │ - chunk_size=256 tokens                             │    │
│  │ - metadata: professor_id, institution               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Redis Cache:                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ embedding:{text_hash} → vector (24h TTL)            │    │
│  │ llm_response:{prompt_hash} → response (24h TTL)     │    │
│  │ context:{thread_id} → serialized context (1h TTL)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Chunking Strategy

```python
class DocumentChunker:
    """Chunk documents for embedding."""
    
    CHUNK_SIZE = 512  # tokens
    CHUNK_OVERLAP = 50  # tokens
    
    def chunk_cv(self, cv: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk CV by section."""
        chunks = []
        
        # Each section becomes a chunk
        for section in ["education", "experience", "publications", "skills"]:
            if section in cv:
                content = json.dumps(cv[section])
                chunks.append({
                    "text": content,
                    "section": section,
                    "metadata": {"type": "cv_section"},
                })
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap."""
        enc = encoding_for_model("text-embedding-3-small")
        tokens = enc.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
            chunk_tokens = tokens[i:i + self.CHUNK_SIZE]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
```

### 11.3 Query Rewriting

```python
class QueryRewriter:
    """Rewrite queries for better retrieval."""
    
    async def rewrite_for_retrieval(
        self,
        query: str,
        context: OrchestrationContext,
    ) -> str:
        """Expand query with context for better matches."""
        
        # Simple expansion
        expansions = []
        
        # Add user research interests
        if context.user.research_interests:
            expansions.append(
                f"Related to: {', '.join(context.user.research_interests[:3])}"
            )
        
        # Add professor context if relevant
        if "professor" in query.lower() and context.professor:
            expansions.append(
                f"Professor research: {', '.join(context.professor.research_areas[:3])}"
            )
        
        if expansions:
            return f"{query}\n\nContext: {' '.join(expansions)}"
        
        return query
```

---

## 12. Moderation & Malicious Behavior Defense

### 12.1 Threat Model

| Threat | Vector | Impact | Mitigation |
|--------|--------|--------|------------|
| **Prompt Injection** | User message | Execute unintended actions | Input sanitization, output validation |
| **Data Exfiltration** | Tool abuse | Leak user data | Tool output limits, audit logging |
| **Jailbreak** | Crafted prompts | Bypass safety | Multi-layer moderation |
| **Memory Poisoning** | Malicious preferences | Corrupt future responses | Memory validation, confidence thresholds |
| **DoS via Tokens** | Long inputs | Cost exhaustion | Input length limits, rate limiting |

### 12.2 Defense Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Defense In Depth                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Input Validation                                   │
│  ├─ Length limits (max 10,000 chars)                        │
│  ├─ Character filtering (control chars)                     │
│  └─ Pattern blocking (injection markers)                    │
│                                                              │
│  Layer 2: Pattern Matching (ModerationAgent)                │
│  ├─ Blocked keywords (BLOCKED_PATTERNS)                     │
│  ├─ Warning keywords (WARNING_PATTERNS)                     │
│  └─ Academic misconduct patterns                            │
│                                                              │
│  Layer 3: LLM Moderation                                    │
│  ├─ Content analysis for nuanced violations                 │
│  ├─ Confidence scoring                                      │
│  └─ Category classification                                 │
│                                                              │
│  Layer 4: Output Validation                                 │
│  ├─ PII detection in responses                              │
│  ├─ Length limits on outputs                                │
│  └─ Consistency checks                                      │
│                                                              │
│  Layer 5: Audit & Alerting                                  │
│  ├─ Log all moderation decisions                            │
│  ├─ Alert on repeated violations                            │
│  └─ Manual review queue                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 Memory Store Protection

```python
class MemoryModerator:
    """Prevent malicious content in memory store."""
    
    async def validate_memory(
        self,
        content: str,
        memory_type: str,
    ) -> Tuple[bool, Optional[str]]:
        """Validate memory before storage."""
        
        # 1. Check for forbidden patterns
        if not is_safe_to_store(content):
            return False, "Content contains restricted information"
        
        # 2. Check for injection attempts
        injection_patterns = [
            r"ignore.*instructions",
            r"you are now",
            r"act as",
            r"system:\s*",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, "Potential injection attempt detected"
        
        # 3. Length check
        if len(content) > 1000:
            return False, "Memory content too long"
        
        # 4. LLM safety check for high-risk types
        if memory_type in ["instruction", "guardrail"]:
            mod_result = await self.moderation_agent.check_content(
                content, "memory"
            )
            if not mod_result.is_safe:
                return False, mod_result.message
        
        return True, None
```

---

## 13. Credits, Usage Metering & Policy

### 13.1 Credit Computation

**Model Pricing** (per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $5.00 | $15.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| text-embedding-3-small | $0.02 | $0.00 |

**Job Cost Estimation**:

| Job Type | Typical Input | Typical Output | Est. Cost |
|----------|--------------|----------------|-----------|
| email_generate | 3,000 | 800 | $0.0009 |
| email_review | 4,000 | 1,200 | $0.0012 |
| cv_generate | 5,000 | 3,000 | $0.0026 |
| cv_review | 6,000 | 2,000 | $0.0021 |
| alignment_evaluate | 4,000 | 1,000 | $0.0012 |
| chat_agentic | 8,000 | 2,000 | $0.0024 |
| chat_direct | 1,500 | 500 | $0.0005 |

### 13.2 Credit Check Flow

```python
class CreditService:
    """Credit management service."""
    
    async def check_and_reserve(
        self,
        student_id: int,
        job_type: str,
        estimated_cost: Decimal,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check credits and reserve for job.
        
        Returns (success, error_message)
        """
        # Get current balance
        balance = await self.get_balance(student_id)
        
        # Check sufficient
        if balance.remaining < estimated_cost:
            if balance.remaining <= 0:
                return False, "Credits depleted. Please purchase more credits."
            else:
                return False, f"Insufficient credits. Need {estimated_cost}, have {balance.remaining}"
        
        # Reserve (soft hold)
        await self.reserve_credits(student_id, estimated_cost)
        
        return True, None
    
    async def finalize_job_cost(
        self,
        student_id: int,
        job_id: int,
        actual_cost: Decimal,
    ) -> None:
        """Finalize job cost after completion."""
        
        # Release reservation
        await self.release_reservation(student_id, job_id)
        
        # Deduct actual cost
        await self.deduct_credits(student_id, actual_cost, job_id)
        
        # Check low balance warning
        balance = await self.get_balance(student_id)
        if balance.remaining < balance.total * Decimal("0.1"):
            await self.event_service.notify_credits_low(
                student_id, 
                float(balance.remaining),
                float(balance.total * Decimal("0.1"))
            )
```

### 13.3 Retry & Failure Policy

| Scenario | Credit Policy |
|----------|--------------|
| Job succeeds | Charge actual cost |
| Job fails (our fault) | No charge |
| Job fails (user input) | No charge for first 2 retries |
| User cancels mid-job | Charge for completed portion |
| Timeout | No charge |
| Multi-step: partial success | Charge for completed steps |

```python
class CreditPolicy:
    """Credit policy for various scenarios."""
    
    MAX_FREE_RETRIES = 2
    
    async def handle_job_failure(
        self,
        job_id: int,
        failure_reason: str,
        is_user_fault: bool,
    ) -> bool:
        """Handle credits for failed job. Returns True if charged."""
        
        job = await self.db.get_job(job_id)
        
        # Get retry count
        retry_count = await self.db.client.aijob.count(
            where={
                "parent_job_id": job.parent_job_id or job.id,
                "status": "failed",
            }
        )
        
        # Our fault - never charge
        if not is_user_fault:
            await self.release_reservation(job.student_id, job_id)
            return False
        
        # User fault - free retries
        if retry_count <= self.MAX_FREE_RETRIES:
            await self.release_reservation(job.student_id, job_id)
            return False
        
        # Beyond free retries - charge
        return await self.finalize_job_cost(
            job.student_id,
            job_id,
            job.cost_total or Decimal("0"),
        )
```

---

## 14. Email Workflow Intelligence

### 14.1 Professor Reply Classification

```python
class ReplyClassifier:
    """Classify professor reply intent and sentiment."""
    
    REPLY_TYPES = [
        "positive_interest",      # Wants to discuss further
        "request_more_info",      # Needs more details
        "schedule_meeting",       # Wants to meet
        "polite_decline",         # Not interested, polite
        "direct_decline",         # Clear no
        "out_of_office",          # Auto-reply
        "forwarded",              # Forwarded to someone else
        "unclear",                # Needs interpretation
    ]
    
    async def classify_reply(
        self,
        reply_body: str,
        original_email: str,
    ) -> Dict[str, Any]:
        """
        Classify reply and extract actionable info.
        
        Returns:
        {
            "type": "positive_interest",
            "sentiment": "positive",  # positive, neutral, negative
            "confidence": 0.85,
            "next_steps": ["Schedule call", "Prepare questions"],
            "key_points": ["Interested in ML work", "Available next week"],
            "suggested_response": "Thank them and propose meeting times",
        }
        """
        prompt = f"""Classify this professor's reply to a student outreach email.

Original email (excerpt):
{original_email[:500]}

Professor's reply:
{reply_body}

Analyze and return JSON with:
- type: one of {self.REPLY_TYPES}
- sentiment: positive, neutral, or negative
- confidence: 0.0 to 1.0
- next_steps: list of recommended actions
- key_points: key information from reply
- suggested_response: how student should respond
"""
        
        response = await self.llm.get_response(
            messages=[{"role": "user", "content": prompt}],
            response_format="json_object",
            temperature=0,
        )
        
        return response.get("output", {})
```

### 14.2 Reminder Logic

```python
class ReminderEngine:
    """Generate contextual follow-up reminders."""
    
    REMINDER_THRESHOLDS = {
        1: timedelta(days=7),   # First reminder after 7 days
        2: timedelta(days=14),  # Second reminder after 14 days
        3: timedelta(days=21),  # Final reminder after 21 days
    }
    
    TONE_PROGRESSION = {
        1: "friendly",
        2: "professional",
        3: "formal_final",
    }
    
    async def should_send_reminder(
        self,
        request_id: int,
        reminder_number: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if reminder is appropriate.
        
        Returns (should_send, reason)
        """
        email = await self.db.get_email(request_id)
        
        # Already sent this reminder
        if getattr(email, f"reminder_{reminder_number}_sent"):
            return False, "Reminder already sent"
        
        # Professor already replied
        if email.professor_replied:
            return False, "Professor has replied"
        
        # Check time since last contact
        last_contact = email.main_sent_at
        if reminder_number > 1:
            for i in range(reminder_number - 1, 0, -1):
                sent_at = getattr(email, f"reminder_{i}_sent_at")
                if sent_at:
                    last_contact = sent_at
                    break
        
        threshold = self.REMINDER_THRESHOLDS[reminder_number]
        if datetime.utcnow() - last_contact < threshold:
            days_remaining = (threshold - (datetime.utcnow() - last_contact)).days
            return False, f"Too soon. Wait {days_remaining} more days."
        
        return True, None
    
    async def generate_reminder(
        self,
        request_id: int,
        reminder_number: int,
    ) -> Dict[str, Any]:
        """Generate reminder with appropriate tone."""
        
        # Get context
        email = await self.db.get_email(request_id)
        request = await self.db.get_funding_request(request_id)
        
        tone = self.TONE_PROGRESSION[reminder_number]
        
        prompt = f"""Generate reminder #{reminder_number} email.

Original email subject: {email.main_email_subject}
Days since last contact: {(datetime.utcnow() - email.main_sent_at).days}
Research connection: {request.research_connection}

Tone: {tone}
{"This is the FINAL reminder. Be polite but clear this is last follow-up." if reminder_number == 3 else ""}

Generate:
- subject (reference original)
- body (brief, remind of interest, request response)
"""
        
        return await self.email_engine.generate(
            user_id=str(request.student_id),
            sender_detail=await self._get_sender_detail(request.student_id),
            recipient_detail=await self._get_recipient_detail(request.professor_id),
            tone=tone,
            generation_type="from_scratch",
            optimization_context={
                "type": "reminder",
                "reminder_number": reminder_number,
                "original_subject": email.main_email_subject,
            }
        )
```

### 14.3 Frontend vs Backend Communication

| Event | Frontend (UX) | Backend (Platform) |
|-------|--------------|-------------------|
| Email generated | Show preview, enable edit | - |
| Email applied | Confirm success toast | Webhook: `email.applied` |
| Reminder appropriate | Show "Send Reminder" button | - |
| Professor replied | Show notification, classify | Webhook: `email.replied` |
| Reply positive | Show celebration, next steps | Webhook: `reply.positive` |
| Reply negative | Show empathy, suggest alternatives | Webhook: `reply.negative` |
| Credit low | Show warning badge | Webhook: `credits.low` |

---

## 15. Scalability & Reliability

### 15.1 Concurrency Strategy

```python
# Current settings (config.py)
JOB_CONCURRENCY = 10  # Max concurrent jobs per instance

# Recommended scaling
CONCURRENCY_CONFIG = {
    "max_concurrent_jobs": 10,
    "max_concurrent_llm_calls": 5,
    "max_concurrent_tool_executions": 3,
    "connection_pool_size": 20,
    "redis_pool_size": 10,
}
```

**Job Queue Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Job Queue Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐    ┌───────────────┐                     │
│  │   API Pod 1   │    │   API Pod 2   │                     │
│  └───────┬───────┘    └───────┬───────┘                     │
│          │                    │                              │
│          └────────┬───────────┘                              │
│                   ▼                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Redis Queue                         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │    │
│  │  │ high    │ │ default │ │ low     │              │    │
│  │  │ (chat)  │ │ (jobs)  │ │ (batch) │              │    │
│  │  └─────────┘ └─────────┘ └─────────┘              │    │
│  └─────────────────────────────────────────────────────┘    │
│                   │                                          │
│          ┌───────┴───────┐                                  │
│          ▼               ▼                                  │
│  ┌───────────────┐ ┌───────────────┐                       │
│  │  Worker Pod 1 │ │  Worker Pod 2 │                       │
│  │  (5 threads)  │ │  (5 threads)  │                       │
│  └───────────────┘ └───────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 15.2 Timeout Configuration

```python
TIMEOUT_CONFIG = {
    # API timeouts
    "http_request_timeout": 30,
    "sse_keepalive_interval": 30,
    
    # LLM timeouts
    "llm_generation_timeout": 60,
    "llm_embedding_timeout": 10,
    
    # Tool timeouts
    "tool_execution_timeout": 60,
    "latex_compilation_timeout": 120,
    
    # Job timeouts
    "job_max_duration": 300,  # 5 minutes
    "job_stuck_threshold": 600,  # 10 minutes
}
```

### 15.3 Idempotency

```python
class IdempotencyService:
    """Ensure operations are idempotent."""
    
    async def get_or_create_job(
        self,
        idempotency_key: str,
        create_func: Callable,
    ) -> Tuple[int, bool]:
        """
        Get existing job or create new one.
        
        Returns (job_id, was_created)
        """
        # Check for existing
        existing = await self.db.client.aijob.find_first(
            where={"trace_id": idempotency_key}
        )
        
        if existing:
            return existing.id, False
        
        # Create with idempotency key
        job = await create_func()
        await self.db.client.aijob.update(
            where={"id": job.id},
            data={"trace_id": idempotency_key}
        )
        
        return job.id, True
```

### 15.4 Observability Requirements

**Metrics to Track**:

| Metric | Type | Purpose |
|--------|------|---------|
| `dana_request_duration_seconds` | Histogram | API latency |
| `dana_tokens_total` | Counter | Token usage |
| `dana_job_status` | Gauge | Job states |
| `dana_tool_calls_total` | Counter | Tool usage |
| `dana_cache_hits_total` | Counter | Cache efficiency |
| `dana_errors_total` | Counter | Error rates |
| `dana_credits_used_total` | Counter | Credit consumption |

**Structured Log Format**:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "info",
  "service": "dana-api",
  "trace_id": "abc123",
  "span_id": "def456",
  "student_id": 123,
  "thread_id": 456,
  "job_id": 789,
  "message": "Tool execution completed",
  "tool": "email_generate",
  "duration_ms": 1234,
  "tokens": {"input": 2500, "output": 800},
  "success": true
}
```

---

## 16. "Anything Missing" Sweep

### 16.1 Unhandled Edge Cases

| Edge Case | Current Handling | Required Handling |
|-----------|-----------------|-------------------|
| Tool fails mid-stream | ❌ None | Retry or graceful degradation |
| User changes mind mid-job | ❌ None | Job cancellation API |
| Conflicting instructions in history | ❌ None | Most recent wins + warn |
| Repeated uploads / versioning | ⚠️ Partial | Full version history |
| Multi-language requests | ❌ None | Language detection + routing |
| Partial completions | ❌ None | Checkpoint & resume |

### 16.2 Tool Failure Recovery

```python
async def execute_tool_with_recovery(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    context: OrchestrationContext,
    max_retries: int = 2,
) -> ToolResult:
    """Execute tool with recovery strategies."""
    
    for attempt in range(max_retries + 1):
        try:
            result = await self.tool_registry.execute(
                name=tool_name,
                arguments=arguments,
                context=context,
            )
            return result
            
        except TimeoutError:
            if attempt < max_retries:
                # Retry with extended timeout
                await asyncio.sleep(2 ** attempt)
                continue
            
            # Return degraded response
            return ToolResult(
                success=False,
                error="Operation timed out",
                data={"suggestion": "Try again with a simpler request"},
            )
            
        except ValidationError as e:
            # Don't retry validation errors
            return ToolResult(
                success=False,
                error=str(e),
                data={"suggestion": "Check input and try again"},
            )
            
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            
            return ToolResult(
                success=False,
                error=str(e),
            )
```

### 16.3 User Intent Change Detection

```python
class IntentChangeDetector:
    """Detect when user changes their mind."""
    
    CANCELLATION_PATTERNS = [
        r"never\s*mind",
        r"cancel\s*(that|this)",
        r"stop",
        r"don't\s*bother",
        r"forget\s*(it|that)",
        r"actually,?\s*(no|wait)",
    ]
    
    MODIFICATION_PATTERNS = [
        r"actually,?\s*(i|let's|can)",
        r"wait,?\s*(i|let's|can)",
        r"instead,?\s*",
        r"change\s*(that|this)\s*to",
    ]
    
    def detect_intent_change(
        self,
        current_message: str,
        previous_message: str,
    ) -> Optional[str]:
        """
        Detect if user is changing their intent.
        
        Returns: "cancel", "modify", or None
        """
        msg_lower = current_message.lower()
        
        for pattern in self.CANCELLATION_PATTERNS:
            if re.search(pattern, msg_lower):
                return "cancel"
        
        for pattern in self.MODIFICATION_PATTERNS:
            if re.search(pattern, msg_lower):
                return "modify"
        
        return None
```

### 16.4 Conflicting Instructions Resolution

```python
class InstructionResolver:
    """Resolve conflicting instructions in thread history."""
    
    async def resolve_conflicts(
        self,
        memories: List[Memory],
        recent_messages: List[Message],
    ) -> List[Memory]:
        """
        Resolve conflicts using recency + explicitness.
        
        Rules:
        1. Explicit > Inferred
        2. Recent > Old
        3. User source > System source
        """
        # Group by topic
        by_topic = defaultdict(list)
        
        for m in memories:
            topic = self._extract_topic(m.content)
            by_topic[topic].append(m)
        
        resolved = []
        
        for topic, items in by_topic.items():
            if len(items) == 1:
                resolved.append(items[0])
                continue
            
            # Sort by priority
            sorted_items = sorted(
                items,
                key=lambda m: (
                    m.source == "user",  # User source preferred
                    m.confidence,        # Higher confidence
                    m.created_at,        # More recent
                ),
                reverse=True,
            )
            
            # Keep top item
            resolved.append(sorted_items[0])
            
            # Deactivate others
            for item in sorted_items[1:]:
                await self.memory_agent.forget(item.id)
        
        return resolved
```

### 16.5 Multi-Language Support

```python
class LanguageHandler:
    """Handle multi-language requests."""
    
    SUPPORTED_LANGUAGES = ["en", "fa", "zh", "ar", "es", "fr"]
    
    async def detect_language(self, text: str) -> str:
        """Detect input language."""
        # Use simple heuristics or langdetect
        from langdetect import detect
        try:
            return detect(text)
        except:
            return "en"
    
    async def should_translate(
        self,
        input_lang: str,
        target_audience: str,
    ) -> bool:
        """Determine if translation is needed."""
        # Academic emails should be in English
        if target_audience == "professor" and input_lang != "en":
            return True
        return False
    
    async def adapt_response(
        self,
        response: str,
        user_lang: str,
    ) -> str:
        """Adapt response to user's language."""
        if user_lang == "en":
            return response
        
        # For non-English users, add note about English output
        if user_lang in ["fa", "ar"]:  # RTL languages
            return f"{response}\n\n(Note: Professional emails should be in English)"
        
        return response
```

### 16.6 Checkpoint & Resume

```python
class CheckpointService:
    """Checkpoint long-running jobs for resumption."""
    
    async def create_checkpoint(
        self,
        job_id: int,
        checkpoint_data: Dict[str, Any],
    ) -> str:
        """Save checkpoint for job."""
        checkpoint_id = f"checkpoint_{job_id}_{datetime.utcnow().timestamp()}"
        
        await self.redis.setex(
            checkpoint_id,
            timedelta(hours=24),
            json.dumps(checkpoint_data),
        )
        
        await self.db.client.aijob.update(
            where={"id": job_id},
            data={"input_payload": json.dumps({"checkpoint": checkpoint_id})}
        )
        
        return checkpoint_id
    
    async def resume_from_checkpoint(
        self,
        job_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Resume job from last checkpoint."""
        job = await self.db.get_job(job_id)
        
        if not job.input_payload:
            return None
        
        payload = json.loads(job.input_payload)
        checkpoint_id = payload.get("checkpoint")
        
        if not checkpoint_id:
            return None
        
        checkpoint_data = await self.redis.get(checkpoint_id)
        
        if checkpoint_data:
            return json.loads(checkpoint_data)
        
        return None
```

---

## Summary of Gaps & Priorities

### Critical (Must Fix Before Launch)

1. **Token-based conversation summarization** - Currently message-count
2. **Suggestion format update** - Need `{title, prompt}` structure
3. **Credit check integration** - Mock implementation
4. **Tool failure recovery** - No retry/fallback
5. **Input sanitization** - Limited injection defense

### High Priority (Launch + 1 Week)

1. **Intent router expansion** - Missing patterns
2. **Reminder generation** - Not implemented
3. **Reply classification** - Not implemented
4. **Document bundling** - Not implemented
5. **Metrics collection** - No observability

### Medium Priority (Launch + 1 Month)

1. **Qdrant integration** - Embeddings in MySQL blob
2. **Multi-language support** - English only
3. **Checkpoint/resume** - Not implemented
4. **Workflow pattern learning** - Static sequences only
5. **Version history** - No document versioning

### Nice to Have (Future)

1. **A/B testing for prompts**
2. **User feedback collection**
3. **Automatic prompt optimization**
4. **Cross-student analytics**
5. **Professor response prediction**

---

## Action Items

See [docs/ACTION_ITEMS.md](./ACTION_ITEMS.md) for prioritized implementation plan.


