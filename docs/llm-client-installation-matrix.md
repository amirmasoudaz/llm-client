# llm-client Installation Matrix

This document defines the supported installation shapes for the standalone
`llm-client` package.

## Supported Python Versions

- Python `3.10`
- Python `3.11`
- Python `3.12`

These versions are validated in CI for installation and package smoke checks.

## Base Install

Use the base package when you want the core runtime, OpenAI provider path,
engine, agent loop, tools, content model, cache APIs, context/memory APIs,
observability, and benchmarks.

```bash
pip install llm-client
```

Editable local install:

```bash
pip install -e .
```

## Optional Extras

### Anthropic

```bash
pip install "llm-client[anthropic]"
```

### Google

```bash
pip install "llm-client[google]"
```

### PostgreSQL Adaptor

```bash
pip install "llm-client[postgres]"
```

### MySQL Adaptor

```bash
pip install "llm-client[mysql]"
```

### Redis Adaptor

```bash
pip install "llm-client[redis]"
```

### Qdrant Adaptor

```bash
pip install "llm-client[qdrant]"
```

### Adaptor Bundle

```bash
pip install "llm-client[adapters]"
```

### Telemetry

```bash
pip install "llm-client[telemetry]"
```

### Performance

```bash
pip install "llm-client[performance]"
```

### Server / FastAPI

```bash
pip install "llm-client[server]"
```

### Development

```bash
pip install -e ".[dev]"
```

### Full Feature Set

```bash
pip install -e ".[anthropic,google,postgres,mysql,redis,qdrant,telemetry,performance,server,dev]"
```

Or:

```bash
pip install -e ".[all]"
```

## CI Validation Matrix

The release/readiness workflows validate these shapes:

- base install on Python `3.10`, `3.11`, and `3.12`
- full-feature install on Python `3.11`
- live cookbook entrypoint validation on Python `3.11` with credential-less skip allowance
- wheel and sdist build verification on Python `3.11`

## Notes

- The base install intentionally includes the core cache/runtime dependencies
  required by the current stable `llm_client.cache` namespace.
- Provider-specific SDKs for Anthropic and Google remain optional extras.
- Service adaptor integrations for PostgreSQL, MySQL, Redis, and Qdrant are
  installable by extras.
- Observability integrations such as OpenTelemetry and Prometheus remain
  optional extras.
- A reusable local service stack is available at
  [docker-compose.llm-client-services.yml](/home/namiral/Projects/Packages/llm-client-v1/docker-compose.llm-client-services.yml)
  for PostgreSQL, Redis, Qdrant, and MySQL-backed package development.
