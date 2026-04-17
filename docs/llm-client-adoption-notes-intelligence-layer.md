# llm-client Adoption Notes: Intelligence Layer

These notes describe how `llm_client` should be used as a reusable
intelligence/runtime layer underneath product systems.

## Intended Role

`llm_client` is the foundational LLM runtime package.

It should serve as:

- the provider abstraction layer
- the execution and tool runtime layer
- the shared model/catalog and capability layer
- the observability and replay substrate

It should not become:

- a business workflow package
- a product orchestration monolith
- a transport/server framework

## Good Consumer Pattern

Higher layers should:

- depend on stable `llm_client` namespaces
- treat the package as the generic kernel
- compose their own product/workflow logic above it

## Poor Consumer Pattern

Avoid:

- importing package internals directly into app logic
- mixing domain workflows into provider/runtime modules
- treating compatibility modules as the long-term API

## Boundary Reminder

Keep these in `llm_client`:

- generic runtime policies
- reusable tool/agent mechanisms
- provider normalization
- shared content/request/result types

Keep these outside:

- product SLAs and approval policy
- domain-specific operators
- application persistence models
- HTTP/UI transports

## Adoption Outcome

When adopted correctly, `llm_client` should reduce duplicate vendor/runtime
logic across repos while leaving domain and product behavior in the consuming
layer.
