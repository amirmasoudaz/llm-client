from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

import os

from agent_runtime import (
    RuntimeKernel,
    JobManager,
    ActionManager,
    PolicyEngine,
    Ledger,
    PostgresPersistedEventBus,
    PluginRegistry,
)
from agent_runtime.storage.postgres import PostgresActionStore, PostgresJobStore, PostgresLedgerWriter

from llm_client import ExecutionEngine
from llm_client.providers.openai import OpenAIProvider
from llm_client.config.agent import AgentConfig
from llm_client.tools.middleware import MiddlewareChain

from intelligence_layer_ops.plugins.platform_plugin import PlatformContextToolsPlugin


@dataclass(frozen=True)
class KernelContainer:
    kernel: RuntimeKernel
    engine: ExecutionEngine


def _build_engine() -> ExecutionEngine:
    mode = os.getenv("IL_LLM_MODE", "openai").lower().strip()

    if mode == "mock":
        from llm_client.providers.base import BaseProvider
        from llm_client.providers.types import CompletionResult, StreamEvent, StreamEventType

        class MockProvider(BaseProvider):
            async def complete(self, messages, **kwargs):
                return CompletionResult(content="(mock) ok", status=200)

            async def stream(self, messages, **kwargs):
                text = "(mock) hello from intelligence layer"
                for ch in text:
                    yield StreamEvent(type=StreamEventType.TOKEN, data=ch)
                yield StreamEvent(type=StreamEventType.DONE, data=CompletionResult(content=text, status=200))

        return ExecutionEngine(provider=MockProvider(model="gpt-5-nano"))

    provider = OpenAIProvider(model=os.getenv("IL_OPENAI_MODEL", "gpt-5-nano"))
    return ExecutionEngine(provider=provider)


async def build_kernel(*, pg_pool: asyncpg.Pool) -> KernelContainer:
    event_bus = PostgresPersistedEventBus(pool=pg_pool)

    job_store = PostgresJobStore(pg_pool)
    action_store = PostgresActionStore(pg_pool)
    ledger_writer = PostgresLedgerWriter(pg_pool)

    job_manager = JobManager(store=job_store, event_bus=event_bus)
    action_manager = ActionManager(store=action_store, job_manager=job_manager, event_bus=event_bus)
    policy_engine = PolicyEngine.default()
    ledger = Ledger(writer=ledger_writer)
    plugins = PluginRegistry()
    plugins.register(PlatformContextToolsPlugin())
    await plugins.load_all()

    engine = _build_engine()

    async def agent_factory(ctx: Any, request: Any) -> Any:
        # Runtime owns budgets/policy/telemetry; keep llm-client tool middleware minimal.
        config = AgentConfig(middleware_chain=MiddlewareChain.minimal_defaults())
        from llm_client import Agent

        tools = plugins.get_tools()
        funding_request_id = None
        if getattr(request, "metadata", None):
            funding_request_id = request.metadata.get("funding_request_id")

        system_message = (
            "You are Dana, the CanApply Intelligence Layer assistant.\n"
            "If you need platform context, call the tool `platform_load_funding_thread_context`.\n"
        )
        if funding_request_id is not None:
            system_message += f"Thread funding_request_id={funding_request_id}\n"

        return Agent(provider=engine.provider, tools=tools, config=config, system_message=system_message)

    kernel = RuntimeKernel(
        job_manager=job_manager,
        action_manager=action_manager,
        policy_engine=policy_engine,
        ledger=ledger,
        event_bus=event_bus,
        plugins=plugins,
        engine=engine,
        agent_factory=agent_factory,
    )
    return KernelContainer(kernel=kernel, engine=engine)
