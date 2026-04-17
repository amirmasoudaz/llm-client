# llm-client 1.2.0 Release Notes

Last updated: 2026-04-16

`llm_client` `1.2.0` is a minor release focused on OpenAI product-surface
expansion, stronger hosted-tool and Realtime ergonomics, broader retrieval
controls, and updated model coverage.

## Added

- OpenAI-specific advanced tool surfaces for:
  - `tool_search`
  - tool namespaces
  - hosted shell and apply-patch continuation outputs
- Broader OpenAI retrieval controls, including typed ranking and attribute
  filtering helpers for file-search and vector-store search workflows.
- Typed vector-store resource controls for expiration policy, chunking
  strategy, and per-file ingestion settings.
- Realtime helper expansion for:
  - conversation item creation
  - audio turn submission
  - collected response output handling
  - MCP tool lifecycle workflows
- GPT-5.4 family model catalog coverage:
  - `gpt-5.4`
  - `gpt-5.4-mini`
  - `gpt-5.4-nano`
  - `gpt-5.4-pro`
- New cookbook examples covering the newer OpenAI advanced surfaces, including
  Realtime conversation lifecycle, tool search/namespaces, Realtime output
  collection, push-to-talk audio turns, and Realtime MCP lifecycle helpers.

## Changed

- The repository root [README.md](/home/namiral/Projects/Packages/llm-client-v1/README.md)
  is now the canonical package reference.
- OpenAI inline file transport now emits documented
  `data:<mime>;base64,...` `file_data` values for inline file blocks in the
  Responses API path.
- Slow live OpenAI cookbook examples were stabilized with higher internal wait
  ceilings and runner timeout overrides where needed.

## Documentation

- Updated the package reference in
  [README.md](/home/namiral/Projects/Packages/llm-client-v1/README.md).
- Updated the package API guide in
  [docs/llm-client-package-api-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-package-api-guide.md).
- Updated the public API map in
  [docs/llm-client-public-api-v1.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-public-api-v1.md).
- Updated the examples guide in
  [docs/llm-client-examples-guide.md](/home/namiral/Projects/Packages/llm-client-v1/docs/llm-client-examples-guide.md).
- Updated the OpenAI capability audits in
  [docs/openai-provider-capability-audit.md](/home/namiral/Projects/Packages/llm-client-v1/docs/openai-provider-capability-audit.md)
  and
  [docs/openai-docs-ledger-completeness-audit-2026-03-31.md](/home/namiral/Projects/Packages/llm-client-v1/docs/openai-docs-ledger-completeness-audit-2026-03-31.md).

## Validation

Full automated validation passed on the release branch state:

```bash
./.venv/bin/pytest -q
```

Result: `349 passed, 3 skipped`

Cookbook validation completed successfully with only expected setup-based
skips:

```bash
set -a && source .env && set +a && \
LLM_CLIENT_EXAMPLE_ALLOW_SKIP=1 \
./.venv/bin/python scripts/ci/run_llm_client_examples.py --subset all --timeout-seconds 120
```

Expected skips remained for environment-dependent examples such as:

- Qdrant-backed examples when Qdrant is unavailable
- PostgreSQL-backed examples when `LLM_CLIENT_EXAMPLE_PG_DSN` is not usable
- MCP approval continuation when approval-loop IDs are not provided
