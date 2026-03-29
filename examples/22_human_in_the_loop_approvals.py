from __future__ import annotations

import asyncio
import json
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from llm_client.engine import ExecutionEngine


APPROVAL_SCOPE = "human-approval-demo"


def _approval_checkpoint(draft: str, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "changes_requested" if not decision["approved"] else "approved",
        "reviewer": decision["reviewer"],
        "requested_changes": decision["notes"],
        "resume_contract": [
            "preserve confirmed facts from the incident packet",
            "apply reviewer changes before any customer-communication language",
            "return a final handoff with explicit internal-only vs customer-ready guidance",
            "do not convert stakeholder preferences into formal approval gates unless explicitly stated",
        ],
        "draft_excerpt": draft[:240],
    }


def _decision_timeline(records: list[Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    ordered = list(reversed(records))
    for index, record in enumerate(ordered, start=1):
        timeline.append(
            {
                "step": index,
                "kind": record.metadata.get("kind"),
                "approved": record.metadata.get("approved"),
                "content": record.content,
            }
        )
    return timeline


async def main() -> None:
    handle = build_live_provider()
    try:
        engine = ExecutionEngine(provider=handle.provider)
        memory = ShortTermMemoryStore()
        incident_packet = {
            "service": "checkout-api",
            "impact": "18-22% of transactions are failing near month-end.",
            "change_context": "payment routing config changed 12 minutes ago",
            "constraints": "leadership wants proactive internal updates before external comms",
        }

        first_prompt = (
            "Prepare an internal incident action plan in markdown with sections: Situation, Proposed Action Plan, "
            "Customer Communication Guidance, Risks, Internal Readiness Guidance. Use only the provided incident packet. "
            "Do not invent formal approval, sign-off, or authorization requirements unless they are explicitly stated in the incident packet."
        )
        first_result = await engine.complete(
            RequestSpec(
                provider=handle.name,
                model=handle.model,
                messages=[
                    Message.system(
                        "You draft action plans for incident response leads. "
                        "Treat stakeholder preferences as sequencing guidance unless the input explicitly defines a formal approval gate. "
                        "Avoid words like approve, approval, sign-off, and authorize unless they are directly supported by the input."
                    ),
                    Message.user(f"{first_prompt}\n\nIncident packet: {incident_packet}"),
                ],
            )
        )
        await memory.write(
            MemoryWrite(
                scope=APPROVAL_SCOPE,
                content=f"Initial draft plan: {first_result.content}",
                metadata={"kind": "draft_plan"},
            )
        )

        approval_decision = {
            "approved": False,
            "reviewer": "incident-commander",
            "notes": (
                "Do not imply external customer communication is ready yet. "
                "Add an internal rollback-evaluation step and make the customer guidance conditional."
            ),
        }
        checkpoint = _approval_checkpoint(first_result.content or "", approval_decision)
        await memory.write(
            MemoryWrite(
                scope=APPROVAL_SCOPE,
                content=approval_decision["notes"],
                metadata={"kind": "approval_feedback", "approved": approval_decision["approved"]},
            )
        )
        await memory.write(
            MemoryWrite(
                scope=APPROVAL_SCOPE,
                content=json.dumps(checkpoint, ensure_ascii=False, sort_keys=True),
                metadata={"kind": "approval_checkpoint", "approved": approval_decision["approved"]},
            )
        )

        second_result = await engine.complete(
            RequestSpec(
                provider=handle.name,
                model=handle.model,
                messages=[
                    Message.system(
                        "Revise the plan after human review. Respect reviewer notes and highlight what changed. "
                        "Do not treat customer communication as approved unless the reviewer says so. "
                        "Do not invent approval requirements or sign-off gates beyond what is explicitly stated in the incident packet or reviewer decision. "
                        "If the packet says leadership wants proactive internal updates before external communication, treat that as sequencing and stakeholder preference, not a hard approval gate. "
                        "Avoid words like approve, approval, sign-off, authorize, or gate unless the incident packet or reviewer note explicitly requires them."
                    ),
                    Message.user(
                        "Incident packet:\n"
                        f"{incident_packet}\n\n"
                        "Original draft:\n"
                        f"{first_result.content}\n\n"
                        "Approval checkpoint:\n"
                        f"{checkpoint}\n\n"
                        "Reviewer decision:\n"
                        f"{approval_decision}\n\n"
                        "Return sections: Final Plan, Human Changes Applied, Remaining Risks, Release/Handoff Status.\n"
                        "Use phrasing like 'internal updates should precede external communications' or "
                        "'external messaging remains conditional/pending internal readiness' unless explicit approval language is provided.\n"
                        "Replace any formal approval phrasing in the original draft with readiness, alignment, sequencing, or customer-readiness wording unless the source input explicitly requires formal approval."
                    ),
                ],
            )
        )
        await memory.write(
            MemoryWrite(
                scope=APPROVAL_SCOPE,
                content=f"Revised final plan: {second_result.content}",
                metadata={"kind": "final_plan"},
            )
        )
        memory_records = await memory.retrieve(MemoryQuery(scope=APPROVAL_SCOPE, limit=5))

        print_heading("Human In The Loop Approvals")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "incident_packet": incident_packet,
                "initial_draft": {
                    "content": first_result.content,
                    "usage": summarize_usage(first_result.usage),
                },
                "approval_decision": approval_decision,
                "approval_checkpoint": checkpoint,
                "memory_records": [
                    {
                        "content": record.content,
                        "metadata": record.metadata,
                    }
                    for record in memory_records
                ],
                "decision_timeline": _decision_timeline(memory_records),
                "final_revision": {
                    "content": second_result.content,
                    "usage": summarize_usage(second_result.usage),
                },
                "showcase_verdict": {
                    "checkpoint_recorded": any(record.metadata.get("kind") == "approval_checkpoint" for record in memory_records),
                    "changes_requested": not approval_decision["approved"],
                    "revision_persisted": any(record.metadata.get("kind") == "final_plan" for record in memory_records),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
