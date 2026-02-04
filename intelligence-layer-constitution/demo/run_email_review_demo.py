import os
import sys

# Allow running as a script: `python demo/run_email_review_demo.py`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from demo.kernel_email_review import DemoKernel


def print_events(run, *, start_index: int = 0) -> int:
    for e in run.events[start_index:]:
        et = e["event_type"]
        if et == "progress":
            p = e["payload"]
            print(f"[progress] {p['percent']:>3}% {p['stage']}: {p['message']}")
        elif et == "action_required":
            p = e["payload"]
            print(f"[action_required] id={p['action_id']} type={p['action_type']} title={p['title']}")
            print(f"  description: {p['description']}")
            print(f"  data: {p.get('data')}")
        elif et == "final":
            print(f"[final] {e['payload'].get('summary')}")
        elif et == "error":
            err = e["payload"]["error"]
            print(f"[error] {err['code']}: {err['message']}")
        else:
            print(f"[{et}] {e}")
    return len(run.events)


def print_outcome_summary(run) -> None:
    out = run.outcomes.get("email_review")
    if not out:
        return
    payload = out.get("payload") or {}
    verdict = payload.get("verdict")
    score = payload.get("overall_score")
    issues = payload.get("issues") or []
    print(f"[outcome] Email.Review verdict={verdict} score={score} issues={len(issues)}")


def main() -> None:
    k = DemoKernel(repo_root=repo_root)

    print("=== Case A: Missing email text -> action_required, then resume ===")
    run = k.submit_email_review(
        thread_id=4512,
        student_id=88,
        funding_request_id=556,
        platform_email_subject=None,
        platform_email_body=None,  # intentionally missing
    )
    k.run_email_review_to_completion(run)
    idx = print_events(run)

    if run.pending_action_id:
        print("\nResolving action with an email body override…\n")
        k.resolve_action_and_resume(
            action_id=run.pending_action_id,
            accepted=True,
            payload={
                "email_subject_override": "Quick question about your lab",
                "email_text_override": "Dear Prof. Demo,\\n\\nI’m reaching out about ...\\n\\nBest,\\nDemo Student",
            },
        )
        idx = print_events(run, start_index=idx)
        print_outcome_summary(run)

    print("\n=== Case B: Platform has email -> review runs in one pass ===")
    run2 = k.submit_email_review(
        thread_id=4513,
        student_id=88,
        funding_request_id=557,
        platform_email_subject="Hello Prof. Demo",
        platform_email_body="Dear Prof. Demo,\\n\\nI’m reaching out to ask about ...\\n\\nBest,\\nDemo Student",
    )
    k.run_email_review_to_completion(run2)
    print_events(run2)
    print_outcome_summary(run2)


if __name__ == "__main__":
    main()
