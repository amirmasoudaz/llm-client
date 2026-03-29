from __future__ import annotations

import importlib


PUBLIC_MODULES = [
    "llm_client.advanced",
    "llm_client.agent",
    "llm_client.compat",
    "llm_client.config",
    "llm_client.content",
    "llm_client.context",
    "llm_client.engine",
    "llm_client.errors",
    "llm_client.model_catalog",
    "llm_client.observability",
    "llm_client.provider_registry",
    "llm_client.providers",
    "llm_client.resilience",
    "llm_client.routing",
    "llm_client.tools",
    "llm_client.types",
    "llm_client.validation",
]


def test_public_modules_define_explicit_all_exports() -> None:
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        assert hasattr(module, "__all__"), module_name
        exported = getattr(module, "__all__")
        assert isinstance(exported, list), module_name
        assert exported, module_name
