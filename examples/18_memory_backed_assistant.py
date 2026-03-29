from __future__ import annotations

import asyncio
import difflib
from dataclasses import dataclass
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.context_assembly import (
    ContextAssemblyRequest,
    ContextSourcePayload,
    ContextSourceRequest,
    MultiSourceContextAssembler,
)
from llm_client.context_planning import (
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
    TieredTrimmingStrategy,
)
from llm_client.engine import ExecutionEngine
from llm_client.memory import InMemorySummaryStore, MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from llm_client.summarization import LLMSummarizer


THREAD_SCOPE = "assistant:enterprise-escalation"


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"


class _ServiceBriefSource:
    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [
            _Entry(
                "system",
                "Service profile: checkout-api handles payment completion, webhook fanout, and export handoff dependencies.",
                entry_type="service_profile",
            ),
            _Entry(
                "system",
                "Operational constraint: finance and compliance customers are highly sensitive to export or settlement delays near month-end.",
                entry_type="service_profile",
            ),
        ]
        return ContextSourcePayload(
            source_name="service_brief",
            entries=entries,
            summary="Service source: checkout-api is payment-critical and tied to high-sensitivity downstream workflows.",
            metadata={"kind": "service"},
        )


class _StakeholderExpectationsSource:
    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [
            _Entry(
                "system",
                "Stakeholder expectation: leadership wants no-surprises updates before any deployment or customer-facing escalation.",
                entry_type="stakeholder_note",
            ),
            _Entry(
                "system",
                "Communication rule: customer updates should acknowledge impact, give one concrete next action, and avoid speculative root-cause claims.",
                entry_type="stakeholder_note",
            ),
        ]
        return ContextSourcePayload(
            source_name="stakeholder_expectations",
            entries=entries,
            summary="Stakeholder source: communications must be concise, factual, and proactive.",
            metadata={"kind": "stakeholder"},
        )


def _format_entries(entries: list[_Entry]) -> list[dict[str, str]]:
    return [
        {
            "role": entry.role,
            "entry_type": entry.entry_type,
            "content": entry.content,
        }
        for entry in entries
    ]


def _format_memory(records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "scope": record.scope,
            "content": record.content,
            "metadata": record.metadata,
        }
        for record in records
    ]


def _summary_change(previous: str | None, current: str | None) -> dict[str, Any]:
    if previous is None and current is None:
        return {"status": "none", "diff": []}
    if previous is None and current is not None:
        return {"status": "created", "diff": current.splitlines()}
    if previous == current:
        return {"status": "unchanged", "diff": []}
    return {
        "status": "updated",
        "diff": list(
            difflib.unified_diff(
                (previous or "").splitlines(),
                (current or "").splitlines(),
                fromfile="previous",
                tofile="current",
                lineterm="",
            )
        ),
    }


def _build_prompt(plan, question: str, *, deliverable: str) -> str:
    selected_context = "\n".join(f"- {entry.content}" for entry in plan.entries)
    memory_context = "\n".join(f"- {record.content}" for record in plan.memory) or "- None"
    summary_context = plan.summary or "None"
    return (
        "You are a senior enterprise incident assistant. Use only the supplied context.\n\n"
        f"Deliverable:\n{deliverable}\n\n"
        f"Current request:\n{question}\n\n"
        f"Selected context:\n{selected_context}\n\n"
        f"Retrieved memory:\n{memory_context}\n\n"
        f"Summary context:\n{summary_context}\n\n"
        "Requirements:\n"
        "1. Be concise and operationally useful.\n"
        "2. Separate confirmed facts from recommendations.\n"
        "3. Do not invent owners or causes not supported by the context.\n"
        "4. Do not imply approval gates or decisions unless explicitly stated in the context.\n"
        "5. If the context expresses a preference for proactive leadership updates, describe it as a communication preference, not an approval requirement.\n"
    )


async def _answer(engine: ExecutionEngine, provider_name: str, model_name: str, prompt: str):  # type: ignore[no-untyped-def]
    return await engine.complete(
        RequestSpec(
            provider=provider_name,
            model=model_name,
            messages=[
                Message.system("You produce executive-ready incident guidance with explicit evidence boundaries."),
                Message.user(prompt),
            ],
        )
    )


async def main() -> None:
    handle = build_live_provider()
    try:
        engine = ExecutionEngine(provider=handle.provider)
        memory = ShortTermMemoryStore()
        summaries = InMemorySummaryStore()

        await memory.write(
            MemoryWrite(
                content="Previous export incident showed audit-log fanout can create queue backpressure without immediate data loss.",
                scope=THREAD_SCOPE,
                metadata={"kind": "prior_incident"},
            )
        )
        await memory.write(
            MemoryWrite(
                content="Leadership prefers a no-surprises briefing before customer updates or deployment decisions.",
                scope=THREAD_SCOPE,
                metadata={"kind": "stakeholder_preference"},
            )
        )
        await memory.write(
            MemoryWrite(
                content="Finance workflows near month-end are treated as escalation-sensitive and should bias severity upward.",
                scope=THREAD_SCOPE,
                metadata={"kind": "business_impact"},
            )
        )

        planner = HeuristicContextPlanner(
            trimming_strategy=TieredTrimmingStrategy(tier1_tail=4),
            memory_reader=memory,
            retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope=THREAD_SCOPE, default_limit=4),
            summarization_strategy=LLMSummarizer(engine=engine),
            summary_store=summaries,
        )
        assembler = MultiSourceContextAssembler(
            planner=planner,
            source_loaders=[
                _ServiceBriefSource(),
                _StakeholderExpectationsSource(),
            ],
        )

        thread_entries = [
            _Entry("user", "We have an enterprise escalation on checkout-api and need an incident commander briefing."),
            _Entry("assistant", "What is the current impact and what changed recently?"),
            _Entry(
                "user",
                "Checkout failures affect 18 to 22 percent of transactions. A payment routing config changed 12 minutes ago.",
            ),
            _Entry("assistant", "What symptoms or alerts are we seeing?"),
            _Entry(
                "user",
                "We have 5xx spikes, rising checkout queue lag, and intermittent webhook delays.",
            ),
            _Entry("assistant", "What operating constraints or stakeholder expectations matter most?"),
            _Entry(
                "user",
                "Finance is close to month-end and leadership wants proactive updates before we talk to the customer.",
            ),
        ]

        first_question = "Prepare a 15-minute incident commander briefing with severity, likely cause, immediate actions, owners, and evidence."
        first_pass = await assembler.assemble(
            ContextAssemblyRequest(
                current_message=first_question,
                base_entries=thread_entries,
                source_request=ContextSourceRequest(current_message=first_question, scope=THREAD_SCOPE),
                max_entries=8,
                memory_query=MemoryQuery(scope=THREAD_SCOPE, query=first_question, limit=4),
                summarize_when_truncated=True,
                persist_summary=True,
                summary_scope=THREAD_SCOPE,
                max_memory_entries=4,
                metadata={"scenario": "memory-backed-assistant"},
            )
        )
        first_prompt = _build_prompt(
            first_pass.plan,
            first_question,
            deliverable=(
                "Incident commander briefing for the internal response lead.\n"
                "Format:\n"
                "- Severity: max 3 bullets\n"
                "- Likely Cause: max 3 bullets\n"
                "- Immediate Actions: max 5 bullets\n"
                "- Owners: max 3 bullets\n"
                "- Evidence: max 4 bullets"
            ),
        )
        first_answer = await _answer(engine, handle.name, handle.model, first_prompt)

        await memory.write(
            MemoryWrite(
                content="Current incident framing: sev-1 checkout degradation tied to recent payment routing config change.",
                scope=THREAD_SCOPE,
                metadata={"kind": "session_takeaway"},
            )
        )
        await memory.write(
            MemoryWrite(
                content="Recommended internal priority is deployment freeze plus targeted rollback evaluation before customer communication.",
                scope=THREAD_SCOPE,
                metadata={"kind": "session_takeaway"},
            )
        )

        follow_up_entries = thread_entries + [
            _Entry("assistant", (first_answer.content or "")[:700], entry_type="assistant_answer"),
            _Entry(
                "user",
                "Now draft a customer update and an executive handoff note that stay aligned with the earlier briefing.",
            ),
        ]
        second_question = "Draft both a customer-facing update and a short executive handoff note using the strongest available context."
        second_pass = await assembler.assemble(
            ContextAssemblyRequest(
                current_message=second_question,
                base_entries=follow_up_entries,
                source_request=ContextSourceRequest(current_message=second_question, scope=THREAD_SCOPE),
                max_entries=10,
                memory_query=MemoryQuery(scope=THREAD_SCOPE, query=second_question, limit=4),
                summarize_when_truncated=True,
                persist_summary=True,
                summary_scope=THREAD_SCOPE,
                max_memory_entries=4,
                metadata={"scenario": "memory-backed-assistant"},
            )
        )
        second_prompt = _build_prompt(
            second_pass.plan,
            second_question,
            deliverable=(
                "1. A customer-facing update in markdown with these sections only: Confirmed Facts, Impact, Planned Next Action, Next Update.\n"
                "2. A separate executive handoff note with these sections only: Confirmed Facts, Risks, Proposed Owners, Next Action."
            ),
        )
        second_answer = await _answer(engine, handle.name, handle.model, second_prompt)

        print_heading("Conversation Thread")
        print_json({"entries": _format_entries(follow_up_entries)})

        print_heading("First Assistant Turn")
        first_summary_change = _summary_change(None, first_pass.plan.persistent_summary)
        print_json(
            {
                "selected_entries": _format_entries(first_pass.plan.entries),
                "memory": _format_memory(first_pass.plan.memory),
                "summary": first_pass.plan.summary,
                "persistent_summary": first_pass.plan.persistent_summary,
                "persistent_summary_change": first_summary_change,
                "metadata": first_pass.plan.metadata,
                "sources": [payload.source_name for payload in first_pass.sources],
                "answer": first_answer.content,
                "usage": summarize_usage(first_answer.usage),
            }
        )

        print_heading("Second Assistant Turn")
        second_summary_change = _summary_change(
            first_pass.plan.persistent_summary,
            second_pass.plan.persistent_summary,
        )
        print_json(
            {
                "selected_entries": _format_entries(second_pass.plan.entries),
                "memory": _format_memory(second_pass.plan.memory),
                "summary": second_pass.plan.summary,
                "persistent_summary": second_pass.plan.persistent_summary,
                "persistent_summary_change": second_summary_change,
                "metadata": second_pass.plan.metadata,
                "sources": [payload.source_name for payload in second_pass.sources],
                "answer": second_answer.content,
                "usage": summarize_usage(second_answer.usage),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
