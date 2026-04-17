# llm-client Adoption Notes: Agent Runtime

These notes describe how an external consumer should adopt `llm_client` for
agent-style workflows.

## Recommended Entry Points

Prefer:

- `llm_client.agent`
- `llm_client.engine`
- `llm_client.tools`
- `llm_client.providers`
- `llm_client.types`

Do not build new integrations around:

- `llm_client.client`
- `llm_client.container`
- provider implementation internals

## Recommended Shape

For new agent integrations:

1. choose a provider or provider registry entry
2. normalize execution through `ExecutionEngine`
3. register tools through `ToolRegistry`
4. run multi-turn behavior through the agent layer
5. keep domain workflow policy above the package

## What the Package Owns

- generic tool-call loops
- provider/runtime normalization
- shared execution policies
- generic streaming and structured-output paths
- reusable observability and replay primitives

## What Consumers Own

- domain prompts
- product-specific agent policies
- escalation and approval rules
- app/server integration
- tenant-specific storage and permissions

## Migration Guidance

If adopting from direct SDK usage:

- move provider calls behind `ExecutionEngine`
- replace ad hoc tool schemas with `Tool` or hosted-tool descriptors
- keep application workflow orchestration outside the package

## Success Criteria

An adoption is in good shape when:

- agent execution uses stable namespaces
- tool/runtime behavior is package-native
- domain logic remains outside `llm_client`
