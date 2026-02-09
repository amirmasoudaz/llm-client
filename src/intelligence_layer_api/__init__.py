"""Layer 2 FastAPI adapter (in-repo prototype).

This package is the Layer 2 frontline:
- talks directly to external services (platform DB, etc.)
- exposes the v1 adapter API described in `intelligence-layer-constitution/API-RUNTIME-DESIGN.md`

It must NOT be imported by `llm_client` or `agent_runtime`.
"""

