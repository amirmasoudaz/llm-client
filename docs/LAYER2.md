# Layer 2 (In-Repo Prototype)

Layer 2 is the frontline service:

- Directly connects to external systems (platform MariaDB/MySQL, S3, Qdrant, Gmail, etc.).
- Exposes the v1 adapter HTTP API described in `intelligence-layer-constitution/API-RUNTIME-DESIGN.md`.
- Uses `agent-runtime` (Layer 1) and `llm-client` (Layer 0) but is **not** imported by them.

## Local dependencies

Bring up Postgres + Redis (constitution-aligned):

```bash
docker compose up -d postgres redis
```

## Env vars (minimal)

### Intelligence Layer Postgres

- `IL_PG_DSN` (preferred) or `PG_DSN` (fallback)

Example:

```bash
IL_PG_DSN="postgresql://postgres:postgres@localhost:5432/intelligence_layer"
```

Note: if you already have `PG_DSN` set (used by other parts of this repo), set `IL_PG_DSN` explicitly so Layer 2 doesn’t accidentally connect to the wrong database.

### Platform DB (MariaDB/MySQL)

Layer 2 reads the platform DB directly (staging credentials are OK for now):

- `PLATFORM_DB_HOST`
- `PLATFORM_DB_PORT`
- `PLATFORM_DB_USER`
- `PLATFORM_DB_PASS`
- `PLATFORM_DB_NAME`
- `PLATFORM_DB_MIN`
- `PLATFORM_DB_MAX`

Fallbacks (for convenience) also read `DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME/DB_MIN/DB_MAX`.

## Run the API (dev)

Install:

```bash
pip install -e ".[layer2]"
```

Run:

```bash
uvicorn intelligence_layer_api.app:app --reload --port 8080
```
