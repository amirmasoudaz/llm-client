# Dana AI Copilot - Complete Project Documentation

> **Project**: Intelligence Layer for Academic Outreach AI Assistant  
> **Version**: 0.1.0  
> **Last Updated**: 2026-02-04  
> **Technology Stack**: Python 3.11+, FastAPI, MySQL 8.0, Redis 7, AWS S3, Docker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Agent System](#3-agent-system)
4. [Services Layer](#4-services-layer)
5. [API Layer](#5-api-layer)
6. [LLM Client](#6-llm-client)
7. [Database Schema](#7-database-schema)
8. [Configuration](#8-configuration)
9. [Deployment](#9-deployment)
10. [Tools & Utilities](#10-tools--utilities)

---

## 1. Project Overview

### 1.1 What is Dana AI Copilot?

Dana is a **deep AI copilot** designed to assist graduate students through their academic outreach journey, from building profiles to sending professor emails. It's a comprehensive, multi-agent system that leverages GPT models for intelligent document generation, review, and optimization.

### 1.2 Core Capabilities

- **Conversational AI**: Natural language interaction via chat threads with Server-Sent Events (SSE) streaming
- **Document Generation**: Academic CVs, SOPs, and outreach emails using LaTeX rendering
- **Intelligent Review**: Multi-dimensional feedback with evidence-based scoring (7-dimension scoring system)
- **Professor Alignment**: Semantic matching between student profiles and professor research
- **Onboarding**: Guided setup for Gmail, profile data, and templates
- **Memory System**: Persistent user preferences with semantic search
- **Program Discovery**: Professor and institution recommendations

### 1.3 Technology Stack & Dependencies

```toml
# Core Framework
FastAPI = "^0.109.0"
Uvicorn = "^0.27.0"
Python = "^3.11"

# Database & ORM
Prisma = "^0.12.0" # Type-safe ORM for MySQL

# AI/ML
OpenAI = "^1.12.0"
Tiktoken = "^0.6.0" # Token counting

# Storage & Caching
Redis = "^5.0.0"
Boto3 = "^1.34.0" (AWS SDK)
Blake3 = "^0.4.0" # Fast hashing

# Document Processing
python-docx = "^1.1.0"
PyPDF2 = "^3.0.0"
Markdown = "^3.5.0"

# Async Utilities
Aiofiles = "^23.2.1"
HTTPX = "^0.26.0"
Aiohttp = "^3.9.0"
```

---

## 2. Architecture

### 2.1 System Architecture

Dana implements a **hybrid orchestration architecture** with three processing modes to optimize token efficiency:

```
┌─────────────────────────────────────────┐
│          Frontend/Platform              │
│        (React/Next.js)                  │
└──────────────┬──────────────────────────┘
               │ REST + SSE
               ▼
┌─────────────────────────────────────────┐
│           Dana API Layer                │
│  ┌──────────┬──────────┬──────────┐     │
│  │ Threads  │Documents │  Usage   │     │
│  └──────────┴──────────┴──────────┘     │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│      Orchestration Layer                │
│  ┌──────────────────────────────────┐   │
│  │   Hybrid Router                  │   │
│  │   • DIRECT (200 tokens)          │   │
│  │   • GUIDED (400 tokens)          │   │
│  │   • AGENTIC (2000+ tokens)       │   │
│  └──────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               ▼
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────┐          ┌─────────┐
│ Email   │          │ Resume  │
│ Agent   │          │ Agent   │
└─────────┘          └─────────┘
    ▼                     ▼
┌──────────────────────────────┐
│       Services Layer         │
│  DB │ Storage │ Jobs │ Cache │
└──────────────────────────────┘
```

### 2.2 Processing Modes

**DIRECT Mode** (Most Efficient - ~200 tokens):
- Single tool call, no reasoning
- Pattern-matched requests (e.g., "review my email")
- Template-based or minimal LLM synthesis

**GUIDED Mode** (Moderate - ~400 tokens):
- Predefined tool sequences
- Multi-step workflows with known patterns
- Example: "Create and review an email"

**AGENTIC Mode** (Full ReAct - ~2000+ tokens):
- Complex, ambiguous requests
- Chain-of-Thought reasoning
- Multi-tool iterations (max 5 iterations)

### 2.3 Key Design Principles

1. **Token Efficiency**: 80% of requests use DIRECT/GUIDED modes
2. **Modularity**: Self-contained agents with clear interfaces
3. **Stateless Operations**: Enables horizontal scaling
4. **Observability**: All LLM calls tracked with tokens, cost, trace_id
5. **Production-Grade**: Rate limiting, retries, graceful degradation

---

## 3. Agent System

### 3.1 Agent Architecture

Each agent is self-contained with:
- **Engine**: Core logic (generation, review, optimization)
- **Context**: Prompts, modules, configuration
- **Schemas**: Type-safe input/output via Pydantic
- **Caching**: Redis/PostgreSQL caching for expensive operations

### 3.2 Agent Catalog

#### 3.2.1 Orchestrator Agent
**Location**: [src/agents/orchestrator/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/orchestrator/engine.py)

**Purpose**: Brain of the system - routes requests and coordinates agents

**Key Components**:
- [DanaOrchestrator](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/orchestrator/engine.py#27-677): Main orchestration engine
- `IntentRouter`: Zero-token classification using pattern matching
- `ToolRegistry`: Function calling interface for all tools
- `ContextBuilder`: Assembles context from multiple sources

**Processing Flow**:
```python
async def process_stream(thread_id, message):
    # 1. Build context
    context = await context_builder.build(thread_id, message)
    
    # 2. Route request (zero LLM tokens)
    route = route_request(message, context)
    
    # 3. Process based on mode
    if route.mode == DIRECT:
        await _process_direct(context, route)
    elif route.mode == GUIDED:
        await _process_guided(context, route)
    else:
        await _process_agentic(context, route)
```

**Routing Logic**:
- Pattern matching (regex)
- Complexity analysis (heuristics)
- Model tier selection (fast vs smart)

#### 3.2.2 Email Agent
**Location**: [src/agents/email/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/email/engine.py)

**Purpose**: Generate, review, and optimize professor outreach emails

**Capabilities**:
```python
class EmailEngine:
    async def generate(
        sender_detail, recipient_detail,
        tone: Literal["formal", "friendly", "enthusiastic"],
        tailor_type: List[Tailor],  # match_research_area, match_recent_papers
        avoid, focus  # Topics to avoid/emphasize
    ) -> EmailGenerationResult
    
    async def review(
        email, sender_detail, recipient_detail
    ) -> EmailReviewResult  # 7-dimensional scoring
    
    async def optimize(
        email, optimization_context
    ) -> EmailOptimizationResult
```

**Review Dimensions** (0-10 scale):
1. Subject Quality
2. Research Fit
3. Evidence Quality
4. Tone Appropriateness
5. Length Efficiency
6. Call to Action
7. Overall Strength

**Readiness Levels**:
- `needs_major_revision`: 1.0-4.99
- `needs_minor_revision`: 5.0-6.99
- `strong`: 7.0-8.49
- `excellent`: 8.5-10.0

**Caching**: Blake3 hash of (messages + model) → cached response

#### 3.2.3 Resume/CV Agent
**Location**: [src/agents/resume/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/resume/engine.py)

**Purpose**: Generate, review, and optimize academic CVs with LaTeX rendering

**Capabilities**:
```python
class CVEngine:
    async def generate(
        user_details, additional_context,
        tone: Literal["academic", "industry", "clinical"]
    ) -> CVGenerationResult
    
    async def review(cv, target_context) -> CVReviewResult
    
    async def optimize(
        cv, sections_to_modify, feedback, user_details
    ) -> CVOptimizationResult
    
    def render_latex(cv) -> Tuple[str, str, str]  # tex, bib, cls
    
    async def compile_pdf(cv, out_dir) -> CompilationResult
```

**LaTeX Compilation**:
- Tries engines in order: latexmk → xelatex → pdflatex
- Error extraction and retry logic
- Multi-pass compilation support

**Review Dimensions**:
1. Content Completeness
2. Research Presentation
3. Technical Depth
4. Publication Quality
5. Structure Clarity
6. Target Alignment
7. Overall Strength

#### 3.2.4 Letter Agent
**Location**: [src/agents/letter/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/letter/engine.py)

**Purpose**: Generate and review academic letters (SOPs, motivation letters)

**Key Features**:
- From-scratch generation
- Review with 7-dimension scoring
- Optimization with iterative refinement
- LaTeX rendering with margin control for page fitting
- Automatic page count optimization (1-page target)

**Unique Capabilities**:
```python
async def render(
    letter, compile_pdf=True,
    margin=1.0,  # Starting margin
    min_margin=0.50,  # Minimum allowed
    step=0.05,  # Reduction step
    max_passes=12  # Max optimization attempts
):
    # Iteratively reduces margin to fit content on 1 page
    # Returns PDF with optimal layout
```

#### 3.2.5 Alignment Agent
**Location**: [src/agents/alignment/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/alignment/engine.py)

**Purpose**: Evaluate alignment between student and professor research

**Scoring System**:
```python
# Criteria-based evaluation with weighted categories
ALIGNMENT_CRITERIA = {
    "research_overlap": 35%,      # Topic match
    "methodological_alignment": 25%,  # Approach match
    "career_stage_fit": 15%,      # Experience level
    "publication_match": 15%,     # Research output
    "skills_compatibility": 10%   # Technical skills
}
```

**Output**:
```python
{
    "overall_score": 7.8,  # 0-100 scale
    "label": "GOOD FIT",   # EXCELLENT/GOOD/MODERATE/WEAK/POOR
    "categories": [...],   # Detailed scores per category
    "reasons": [...],      # Top positive/negative signals
    "diagnostics": {       # Confidence metrics
        "high_conf_rate": 0.82,
        "mean_intensity": 0.75
    }
}
```

**Deterministic Scoring**: Two-phase approach
1. LLM generates judgments (Yes/No + intensity)
2. Deterministic formula converts to 0-100 score

#### 3.2.6 Memory Agent
**Location**: [src/agents/memory/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/memory/engine.py)

**Purpose**: Persistent storage and retrieval of user preferences

**Memory Types**:
- `tone`: Email/letter tone preferences
- `do_dont`: Things to include/exclude
- `preference`: General preferences
- `goal`: Academic goals
- `bio`: Background facts
- `instruction`: User instructions
- `guardrail`: Constraints

**Key Features**:
```python
class MemoryAgent:
    async def push(student_id, memory_type, content, ttl_days, embed=True)
    async def pull(student_id, memory_type, query, limit)
    async def forget(memory_id)  # Soft delete
    async def extract_from_conversation(messages, auto_store=True)
```

**Semantic Search**:
- Uses `text-embedding-3-small` (1536 dimensions)
- Cosine similarity matching
- Deduplication via Blake3 hashing

#### 3.2.7 Moderation Agent
**Location**: [src/agents/moderation/engine.py](file:///home/namiral/Projects/CanApply/intelligence-layer/src/agents/moderation/engine.py)

**Purpose**: Content safety and policy compliance

#### 3.2.8 Onboarding Agents
**Location**: `src/agents/onboarding/`

**Sub-agents**:
- `data.py`: Profile data collection
- `gmail.py`: Gmail OAuth integration
- `template.py`: Email template finalization

**Purpose**: Guided user setup flow

#### 3.2.9 Programs Agent
**Location**: `src/agents/programs/engine.py`

**Purpose**: Professor and institution discovery/recommendations

#### 3.2.10 Converter Agent
**Location**: `src/agents/converter/engine.py`

**Purpose**: Convert documents (PDF, DOCX, images) to structured JSON

**Supported Formats**:
- `.txt`, `.pdf`, `.docx`, `.md`
- `.png`, `.jpg`, `.jpeg`, `.webp` (OCR)
- `.tex`

**Two-Stage Pipeline**:
```python
# Stage 1: Parse to raw text
raw_content = await _load_parsed(cache=True)

# Stage 2: LLM conversion to structured JSON
json_output = await converter_model.get_response(
    messages=messages,
    response_format=CVGenerationRespSchema
)
```

---

## 4. Services Layer

### 4.1 Database Service
**Location**: `src/services/db.py`

**Purpose**: Type-safe database operations using Prisma ORM

**Core Operations**:

```python
class DatabaseService:
    # Thread Management
    async def create_thread(funding_request_id, student_id, title)
    async def get_thread(thread_id)
    async def update_thread_status(thread_id, status)
    
    # Message Management  
    async def create_message(thread_id, role, content, message_type)
    async def get_thread_messages(thread_id, limit, before_idx)
    async def get_recent_messages(thread_id, limit=10)
    
    # Job Management
    async def create_job(student_id, job_type, model, ...)
    async def complete_job(job_id, result_payload, usage, trace_id)
    async def fail_job(job_id, error_message, error_code)
    
    # Memory Management
    async def create_memory(student_id, memory_type, content, ...)
    async def get_student_memories(student_id, memory_type, active_only)
    async def deactivate_memory(memory_id)
    
    # Document Management
    async def create_document(student_id, title, document_type, ...)
    async def list_documents(student_id, document_type, status, ...)
    async def get_thread_documents(thread_id)
    
    # Usage Analytics
    async def get_student_usage(student_id, from_date, to_date)
    async def get_job_usage(student_id, from_date, to_date, ...)
```

### 4.2 Storage Service
**Location**: `src/services/storage.py`

**Purpose**: S3 file management with three-tier lifecycle

**File Lifecycles**:
1. **temps**: Temporary files for processing (TTL-based cleanup)
2. **sandbox**: Work-in-progress pending approval  
3. **finals**: Approved/finalized documents

**Operations**:
```python
class StorageService:
    # Temp Files
    async def create_temp(student_id, thread_id, content, extension)
    async def get_temp(key) -> bytes
    async def cleanup_thread_temps(student_id, thread_id)
    
    # Sandbox Files
    async def create_sandbox(student_id, thread_id, content, ...)
    async def promote_to_sandbox(temp_key, student_id, thread_id, ...)
    
    # Final Files
    async def finalize(sandbox_key, student_id, extension)  # Content-addressed
    async def get_final(key) -> bytes
    
    # Document Upload
    async def upload_document(student_id, content, filename, title, ...)
    async def stream_download(key, chunk_size)
```

**S3 Key Structure**:
```
platform/dana/{student_id}/
├── temporary/{thread_id}/{uuid}.{ext}    # TTL cleanup
├── sandbox/{thread_id}/{hash}.{ext}      # WIP files
├── documents/{hash}.{ext}                # Finalized (content-addressed)
└── sources/{hash}.{ext}                  # Uploads (content-addressed)
```

### 4.3 Jobs Service
**Location**: `src/services/jobs.py`

**Purpose**: Background job execution and tracking

**Job Types**:
- `email_generate`, `email_review`, `email_optimize`
- `resume_generate`, `resume_review`, `resume_optimize`
- `letter_generate`, `letter_review`, `letter_optimize`
- `alignment_evaluate`
- `chat_direct`, `chat_guided`, `chat_agentic`

**Job Lifecycle**:
```
queued → running → succeeded/failed/cancelled
```

### 4.4 Events Service
**Location**: `src/services/events.py`

**Purpose**: Server-Sent Events (SSE) streaming for real-time updates

**Event Types**:
```python
class EventType(str, Enum):
    RESPONSE_START = "response_start"
    RESPONSE_TOKEN = "response_token"
    RESPONSE_END = "response_end"
    PROGRESS_UPDATE = "progress_update"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    META_ACTION = "meta_action"
    ERROR = "error"
```

### 4.5 Usage Service
**Location**: `src/services/usage.py`

**Purpose**: Token/cost tracking and credit management

**Metrics Tracked**:
- Token usage (input/output/total)
- Cost (per model tier)
- Request counts
- Cache hit rates

---

## 5. API Layer

### 5.1 Application Structure
**Location**: `src/api/app.py`

**Framework**: FastAPI with CORS and rate limiting middleware

```python
app = FastAPI(
    title="Dana AI Copilot",
    version="0.1.0",
    lifespan=lifespan  # Async startup/shutdown
)

# Middleware
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RateLimitMiddleware)

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(threads.router, prefix="/threads")
app.include_router(documents.router, prefix="/documents")
app.include_router(usage.router, prefix="/usage")
app.include_router(enhance.router, prefix="/ai")
```

### 5.2 API Routes

#### Threads API
**Location**: `src/api/routes/threads.py`

**Endpoints**:
```
POST   /threads                    # Create thread
GET    /threads/{thread_id}         # Get thread
GET    /threads/{thread_id}/messages  # List messages
POST   /threads/{thread_id}/messages  # Send message (SSE stream)
GET    /threads                     # List user threads
PATCH  /threads/{thread_id}         # Update thread  
DELETE /threads/{thread_id}         # Archive thread
```

#### Documents API  
**Location**: `src/api/routes/documents.py`

**Endpoints**:
```
POST   /documents/upload            # Upload document
GET    /documents/{doc_id}          # Get document
GET    /documents                   # List documents
DELETE /documents/{doc_id}          # Delete document
GET    /documents/{doc_id}/download # Download file
```

#### Usage API
**Location**: `src/api/routes/usage.py`

**Endpoints**:
```
GET    /usage/summary              # Usage summary
GET    /usage/jobs                 # Job history
GET    /usage/credits              # Credit status
```

#### AI Enhancement API
**Location**: `src/api/routes/enhance.py`

**Endpoints**:
```
POST   /ai/email/generate          # Generate email
POST   /ai/email/review            # Review email
POST   /ai/email/optimize          # Optimize email
POST   /ai/resume/generate         # Generate CV
POST   /ai/resume/review           # Review CV
POST   /ai/letter/generate         # Generate SOP
POST   /ai/alignment/evaluate      # Alignment check
```

### 5.3 Middleware

#### Rate Limiting
**Location**: `src/api/middleware/rate_limit.py`

**Implementation**: Token bucket algorithm per user
- Requests per minute
- Token budget per minute
- Burst allowance

---

## 6. LLM Client

### 6.1 Overview
**Location**: `src/llm-client/`

**Purpose**: Unified async LLM client with caching, rate limiting, and streaming

**Key Features**:
- Model profiles for GPT-5 family
- Automatic token counting and cost estimation
- Pluggable response caching (filesystem, Qdrant, PostgreSQL + Redis)
- Token-aware rate limiting
- Pusher-based SSE streaming
- Batch processing helpers

### 6.2 Model Profiles

**Available Models**:
```python
# Completions
GPT5 = "gpt-5"           # Smartest, most expensive
GPT5Mini = "gpt-5-mini"  # Balanced
GPT5Nano = "gpt-5-nano"  # Fast, cheapest
GPT5Point1 = "gpt-5.1"   # Latest version
GPT5Point2 = "gpt-5.2"

# Embeddings
TextEmbedding3Large = "text-embedding-3-large"  # 3072 dimensions
TextEmbedding3Small = "text-embedding-3-small"  # 1536 dimensions
```

**Model Profile**:
```python
@dataclass
class ModelProfile:
    key: str                    # "gpt-5-mini"
    model_name: str             # API name
    category: str               # "completions" | "embeddings"
    context_window: int         # 128000
    max_output: int             # 16384
    rate_limits: RateLimits     # tkn_per_min, req_per_min
    usage_costs: UsageCosts     # input/output/cached costs
```

### 6.3 Caching Backends

**Filesystem Cache**:
```python
client = OpenAIClient(
    GPT5Nano,
    cache_backend="fs",
    cache_dir=Path("cache/completions")
)
```

**PostgreSQL + Redis Cache** (Production):
```python
client = OpenAIClient(
    GPT5Nano,
    cache_backend="pg_redis",
    cache_collection="dana_fast",
    pg_dsn="postgresql://...",
    redis_url="redis://localhost:6379/0"
)
```

**Cache Strategy**:
- **L1 (Redis)**: Hot cache with TTL
- **L2 (PostgreSQL)**: Durable storage with optional compression
- **Key**: Blake3 hash of (messages + model + params)

### 6.4 API Usage

**Basic Completion**:
```python
response = await client.get_response(
    messages=[{"role": "user", "content": "Hello"}],
    cache_response=True,
    temperature=0.7
)
# response = {"output": "...", "usage": {...}, "status": 200}
```

**Structured Output**:
```python
from pydantic import BaseModel

class EmailSchema(BaseModel):
    subject: str
    body: str

response = await client.get_response(
    messages=messages,
    response_format=EmailSchema,
    temperature=0
)
email = response["output"]  # Validated EmailSchema instance
```

**Streaming**:
```python
response = await client.get_response(
    messages=messages,
    stream=True,
    channel="my-channel"  # Pusher channel
)
# Sends "new-token" events via Pusher
```

**Embeddings**:
```python
client = OpenAIClient(TextEmbedding3Small)
response = await client.get_response(
    input="Text to embed"
)
embedding = response["output"]  # List[float] of length 1536
```

### 6.5 Rate Limiting

**Token Bucket Implementation**:
```python
async with limiter.limit(tokens=1500, requests=1):
    response = await openai_api_call()
    # Automatically charges output tokens back to bucket
```

**Per-Model Limits**:
- GPT5: 800K tokens/min, 10K req/min
- GPT5Mini: 2M tokens/min, 30K req/min
- GPT5Nano: 2M tokens/min, 30K req/min

---

## 7. Database Schema

### 7.1 Core Tables

#### `chat_threads`
Conversation threads tied to funding requests.

```sql
CREATE TABLE chat_threads (
  id                  BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  funding_request_id  BIGINT UNSIGNED NOT NULL,
  student_id          BIGINT UNSIGNED NOT NULL,
  title               VARCHAR(255),
  summary             TEXT,                -- Context compression
  suggestions         JSON,                -- Cached follow-ups
  status              VARCHAR(20) DEFAULT 'active',
  created_at          TIMESTAMP,
  updated_at          TIMESTAMP,
  
  INDEX idx_student_created (student_id, created_at),
  INDEX idx_request (funding_request_id)
);
```

#### `chat_thread_messages`
Individual messages in threads.

```sql
CREATE TABLE chat_thread_messages (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  thread_id     BIGINT UNSIGNED NOT NULL,
  message_idx   INT UNSIGNED NOT NULL,
  role          VARCHAR(20) NOT NULL,     -- user, assistant, system, tool
  message_type  VARCHAR(20) DEFAULT 'message',
  content       JSON NOT NULL,
  tool_name     VARCHAR(128),
  tool_payload  JSON,
  created_at    TIMESTAMP,
  
  UNIQUE KEY uk_thread_idx (thread_id, message_idx)
);
```

#### `ai_jobs`
Tracks all AI operations.

```sql
CREATE TABLE ai_jobs (
  id              BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  student_id      BIGINT UNSIGNED NOT NULL,
  job_type        VARCHAR(50) NOT NULL,
  status          VARCHAR(20) DEFAULT 'queued',
  progress        TINYINT UNSIGNED DEFAULT 0,
  target_type     VARCHAR(30) NOT NULL,
  target_id       BIGINT UNSIGNED NOT NULL,
  thread_id       BIGINT UNSIGNED,
  
  input_payload   JSON,
  result_payload  JSON,
  
  -- Timing
  created_at      TIMESTAMP,
  started_at      TIMESTAMP NULL,
  finished_at     TIMESTAMP NULL,
  
  -- Model & tracing
  model           VARCHAR(64) NOT NULL,
  trace_id        VARCHAR(64),
  
  -- Token usage
  token_input     INT UNSIGNED DEFAULT 0,
  token_output    INT UNSIGNED,
  token_total     INT UNSIGNED DEFAULT 0,
  
  -- Cost (USD)
  cost_input      DECIMAL(10, 6) DEFAULT 0,
  cost_output     DECIMAL(10, 6),
  cost_total      DECIMAL(10, 6) DEFAULT 0,
  
  INDEX idx_status_created (status, created_at),
  INDEX idx_student_created (student_id, created_at)
);
```

#### `ai_memory`
Long-term user preferences.

```sql
CREATE TABLE ai_memory (
  id            BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  student_id    BIGINT UNSIGNED NOT NULL,
  memory_type   VARCHAR(30) NOT NULL,
  content       TEXT NOT NULL,
  content_hash  CHAR(64),
  source        VARCHAR(20) DEFAULT 'inferred',
  confidence    DECIMAL(4, 3) DEFAULT 0.700,
  embedding     BLOB,                -- Vector (1536 floats)
  is_active     BOOLEAN DEFAULT TRUE,
  expires_at    TIMESTAMP NULL,
  created_at    TIMESTAMP,
  
  INDEX idx_student_type_active (student_id, memory_type, is_active)
);
```

#### `student_documents`
Document storage with processing lifecycle.

```sql
CREATE TABLE student_documents (
  id                      BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  student_id              BIGINT UNSIGNED NOT NULL,
  title                   VARCHAR(255) NOT NULL,
  document_type           VARCHAR(30) NOT NULL,
  
  source_file_path        VARCHAR(1024) NOT NULL,
  exported_pdf_path       VARCHAR(1024),
  
  source_file_hash        CHAR(64) NOT NULL,
  extracted_text_hash     CHAR(64),
  
  upload_status           VARCHAR(30) DEFAULT 'uploaded',
  
  raw_extracted_text      LONGTEXT,
  processed_content       JSON,
  missing_fields          JSON,
  
  created_at              TIMESTAMP,
  updated_at              TIMESTAMP,
  
  INDEX idx_student_type_created (student_id, document_type, created_at)
);
```

### 7.2 Platform Integration Tables (Read-Only)

- `students`: User accounts
- `funding_professors`: Professor database
- `funding_institutes`: Institution information
- `funding_requests`: Funding application requests
- `funding_emails`: Email tracking
- `funding_credentials`: Gmail OAuth tokens

---

## 8. Configuration

### 8.1 Environment Variables
**Location**: `src/config.py`

```python
class Settings:
    # App
    app_env: str                    # dev | prod
    debug: bool
    job_poll_interval_s: float
    job_max_concurrency: int
    
    # Database
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    
    # Redis
    redis_url: str
    
    # Storage
    storage_backend: str            # s3 | local
    s3_region: str
    s3_bucket: str
    
    # LLM
    llm_mode: str                   # live | mock
    llm_review_model: str           # gpt-4o-mini
    llm_revision_model: str
    llm_doc_model: str
    llm_orchestrator_model: str
    
    # Platform Integration
    platform_backend_url: str
    platform_webhook_url: str
    platform_api_key: str
```

### 8.2 Docker Configuration

**docker-compose.yml Services**:
1. `dana-api`: Main FastAPI application
2. `mysql`: MySQL 8.0 database
3. `redis`: Redis 7 cache
4. `prisma-studio`: Database GUI (dev profile)

**Volumes**:
- `mysql-data`: Persistent database
- `redis-data`: Persistent cache
- `dana-data`: Application data

---

## 9. Deployment

### 9.1 Docker Deployment

**Build & Run**:
```bash
docker-compose up -d
```

**Services Exposed**:
- Dana API: `http://localhost:8000`
- MySQL: `localhost:3307`
- Redis: `localhost:6380`
- Prisma Studio: `http://localhost:5555` (dev profile)

### 9.2 Production Dockerfile

**Multi-stage build**:
1. **Builder stage**: Install poetry dependencies
2. **Production stage**: Copy packages, generate Prisma client, run as non-root

**Health Check**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## 10. Tools & Utilities

### 10.1 Async S3 Client
**Location**: `src/tools/async_s3.py`

**Features**:
- Async boto3 wrapper
- Connection pooling (max 64)
- Rate limiting (50 req/s, burst 100)
- Streaming upload/download
- Batch operations

### 10.2 Email Diff
**Location**: `src/tools/email_diff.py`

**Purpose**: Compare email versions and highlight changes

### 10.3 Hash Utility
**Location**: `src/tools/get_hash.py`

**Purpose**: Blake3 hashing for content-addressed storage

---

## Summary

Dana AI Copilot is a production-ready, multi-agent system featuring:

✅ **10 Specialized Agents** with distinct responsibilities  
✅ **Hybrid Orchestration** optimizing 80% of requests to \u003c400 tokens  
✅ **Comprehensive Caching** via PostgreSQL + Redis  
✅ **LaTeX Document Generation** with automatic PDF compilation  
✅ **7-Dimension Review System** with deterministic scoring  
✅ **Semantic Memory** with embedding-based search  
✅ **Real-time Streaming** via Server-Sent Events  
✅ **Type-Safe Database** operations with Prisma ORM  
✅ **Content-Addressed Storage** for deduplication  
✅ **Rate Limiting** with token bucket algorithm  
✅ **Observable** with full token/cost tracking

**Total Lines of Code**: ~15,000+ across all modules  
**Core Agents**: 10  
**API Endpoints**: 20+  
**Database Tables**: 12+  

For detailed implementation of specific components, refer to the source files listed in each section.

