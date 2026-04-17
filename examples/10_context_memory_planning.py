from __future__ import annotations

import asyncio
import difflib
import json
import os
from dataclasses import dataclass
from typing import Any

from llm_client import ExecutionEngine, Message, OpenAIProvider, RequestSpec, load_env
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
from llm_client.memory import InMemorySummaryStore, MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.summarization import LLMSummarizer

load_env()

THREAD_SCOPE = "grant-thread-robotics"


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"


class _ApplicantProfileSource:
    @staticmethod
    async def load(request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [
            _Entry(
                "system",
                "Applicant profile: robotics lab lead focused on safe warehouse automation for small manufacturers.",
                entry_type="profile",
            ),
            _Entry(
                "system",
                "Applicant constraint summary: limited pilot budget, 9-month timeline, strict safety validation requirement.",
                entry_type="profile",
            ),
        ]
        return ContextSourcePayload(
            source_name="applicant_profile",
            entries=entries,
            summary="Profile source: robotics automation applicant with budget, timeline, and safety constraints.",
            metadata={"kind": "profile"},
        )


class _ReviewerBriefSource:
    @staticmethod
    async def load(request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [
            _Entry(
                "system",
                "Reviewer rubric: emphasize measurable impact, deployment feasibility, and safety governance.",
                entry_type="reviewer_note",
            ),
            _Entry(
                "system",
                "Reviewer caution: avoid over-claiming autonomous deployment before pilot validation is complete.",
                entry_type="reviewer_note",
            ),
        ]
        return ContextSourcePayload(
            source_name="reviewer_brief",
            entries=entries,
            summary="Reviewer source: reward measurable impact and realism; penalize unsafe or over-scoped claims.",
            metadata={"kind": "reviewer"},
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


def _build_prompt(plan, question: str) -> str:
    history_block = "\n".join(f"- {entry.content}" for entry in plan.entries)
    memory_block = "\n".join(f"- {record.content}" for record in plan.memory) or "- None"
    summary_block = plan.summary or "None"
    return (
        "You are a senior grant-writing advisor. Use only the supplied context.\n\n"
        f"Question:\n{question}\n\n"
        f"Selected conversation context:\n{history_block}\n\n"
        f"Retrieved memory:\n{memory_block}\n\n"
        f"Summary context:\n{summary_block}\n\n"
        "Produce:\n"
        "1. A concise grant-positioning brief.\n"
        "2. Three strongest strengths.\n"
        "3. Three key risks or gaps.\n"
        "4. A recommended next drafting move.\n"
    )


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


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider_name = "openai"
    provider = OpenAIProvider(model=model_name)
    try:
        engine = ExecutionEngine(provider=provider)
        memory = ShortTermMemoryStore()
        summaries = InMemorySummaryStore()

        await memory.write(
            MemoryWrite(
                content="Applicant is pursuing robotics grant funding for safe warehouse automation pilots.",
                scope=THREAD_SCOPE,
                metadata={"kind": "program"},
            )
        )
        await memory.write(
            MemoryWrite(
                content="Past reviewer feedback praised safety framing but wanted stronger deployment feasibility details.",
                scope=THREAD_SCOPE,
                metadata={"kind": "feedback"},
            )
        )
        await memory.write(
            MemoryWrite(
                content="The pilot budget ceiling is $250k and deployment must complete inside 9 months.",
                scope=THREAD_SCOPE,
                metadata={"kind": "constraint"},
            )
        )

        planner = HeuristicContextPlanner(
            trimming_strategy=TieredTrimmingStrategy(tier1_tail=4),
            memory_reader=memory,
            retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope=THREAD_SCOPE, default_limit=3),
            summarization_strategy=LLMSummarizer(engine=engine),
            summary_store=summaries,
        )
        assembler = MultiSourceContextAssembler(
            planner=planner,
            source_loaders=[
                _ApplicantProfileSource(),
                _ReviewerBriefSource(),
            ],
        )

        thread_entries = [
            _Entry("user", "I am applying for a robotics commercialization grant."),
            _Entry("assistant", "What is the core problem your proposal solves?"),
            _Entry("user", "We reduce warehouse safety incidents during pallet movement."),
            _Entry("assistant", "What constraints do reviewers care about most?"),
            _Entry("user", "Budget realism, deployment timeline, and safety validation."),
            _Entry("assistant", "What proof points do you already have?"),
            _Entry("user", "A small pilot reduced operator intervention by 31%."),
            _Entry("assistant", "What are reviewers still skeptical about?"),
            _Entry("user", "They worry we sound too ambitious on autonomous rollout."),
            _Entry("assistant", "What do you need help with next?"),
            _Entry("user", "I need a compelling problem statement and risk framing."),
        ]

        first_question = "Help me prepare a concise context packet for a grant draft."
        first_pass = await assembler.assemble(
            ContextAssemblyRequest(
                current_message=first_question,
                base_entries=thread_entries,
                source_request=ContextSourceRequest(current_message=first_question, scope=THREAD_SCOPE),
                max_entries=9,
                memory_query=MemoryQuery(scope=THREAD_SCOPE, query=first_question, limit=3),
                summarize_when_truncated=True,
                persist_summary=True,
                summary_scope=THREAD_SCOPE,
                max_memory_entries=3,
                metadata={"scenario": "grant-context-showcase"},
            )
        )

        second_question = (
            "Draft a grant-positioning brief that balances ambition with deployment realism, "
            "using the strongest relevant context."
        )
        follow_up_entries = thread_entries + [
            _Entry("assistant", "Do you want the proposal to emphasize commercialization or research novelty?"),
            _Entry("user", "Commercialization first, but still mention the safety research foundation."),
        ]
        second_pass = await assembler.assemble(
            ContextAssemblyRequest(
                current_message=second_question,
                base_entries=follow_up_entries,
                source_request=ContextSourceRequest(current_message=second_question, scope=THREAD_SCOPE),
                max_entries=9,
                memory_query=MemoryQuery(scope=THREAD_SCOPE, query=second_question, limit=3),
                summarize_when_truncated=True,
                persist_summary=True,
                summary_scope=THREAD_SCOPE,
                max_memory_entries=3,
                metadata={"scenario": "grant-context-showcase"},
            )
        )

        final_prompt = _build_prompt(second_pass.plan, second_question)
        final_result = await engine.complete(
            spec=RequestSpec(
                provider=provider_name,
                model=model_name,
                messages=[
                    Message.system("You write concise, evidence-based grant guidance."),
                    Message.user(final_prompt),
                ],
            )
        )

        usage = (
            {
                "input_tokens": final_result.usage.input_tokens,
                "output_tokens": final_result.usage.output_tokens,
                "total_tokens": final_result.usage.total_tokens,
                "total_cost": final_result.usage.total_cost,
            }
            if final_result.usage is not None
            else {}
        )

        print("\n=== Source Thread ===\n")
        print(json.dumps({"entries": _format_entries(follow_up_entries)}, indent=2, ensure_ascii=False, default=str))

        print("\n=== First Planning Pass ===\n")
        first_summary_change = _summary_change(None, first_pass.plan.persistent_summary)
        print(
            json.dumps(
                {
                    "selected_entries": _format_entries(first_pass.plan.entries),
                    "memory": _format_memory(first_pass.plan.memory),
                    "summary": first_pass.plan.summary,
                    "persistent_summary": first_pass.plan.persistent_summary,
                    "metadata": first_pass.plan.metadata,
                    "persistent_summary_change": first_summary_change,
                    "sources": [payload.source_name for payload in first_pass.sources],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Second Planning Pass ===\n")
        second_summary_change = _summary_change(
            first_pass.plan.persistent_summary,
            second_pass.plan.persistent_summary,
        )
        print(
            json.dumps(
                {
                    "selected_entries": _format_entries(second_pass.plan.entries),
                    "memory": _format_memory(second_pass.plan.memory),
                    "summary": second_pass.plan.summary,
                    "persistent_summary": second_pass.plan.persistent_summary,
                    "metadata": second_pass.plan.metadata,
                    "persistent_summary_change": second_summary_change,
                    "sources": [payload.source_name for payload in second_pass.sources],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Final Live Answer ===\n")
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "prompt_preview": final_prompt[:1200],
                    "status": final_result.status,
                    "usage": usage,
                    "content": final_result.content,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
