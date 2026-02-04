import argparse
import asyncio
import os
import sys
from typing import Any, Dict, Optional

# Ensure repo root is on sys.path so `demo.*` imports work when run as a script.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from demo.kernel_ai_outreach import AIDemoKernel, MockLLM


def _print_event(event: Dict[str, Any]) -> None:
    et = event["event_type"]
    if et == "progress":
        p = event["payload"]
        print(f"[progress] {p['percent']:>3}% {p['stage']}: {p['message']}")
        return
    if et == "token_delta":
        sys.stdout.write(event["payload"]["delta"])
        sys.stdout.flush()
        return
    if et == "action_required":
        p = event["payload"]
        print("\n")
        print(f"[action_required] id={p['action_id']} type={p['action_type']} title={p['title']}")
        print(f"  description: {p['description']}")
        if p.get("proposed_changes"):
            print("  proposed_changes: (platform patch proposal)")
        return
    if et == "final":
        p = event["payload"]
        print("\n")
        print(f"[final] {p.get('summary')}")
        return
    if et == "error":
        err = event["payload"]["error"]
        print(f"[error] {err['code']}: {err['message']}")
        return
    print(f"[{et}] {event}")

def _read_multiline_message() -> Optional[str]:
    """
    Read a user message from stdin.

    Supports pasting multi-line content if you wrap it in either:
    - triple backticks ``` ... ```
    - double quotes " ... "

    We auto-continue reading until the wrapper is closed.
    """
    try:
        first = input("You: ")
    except EOFError:
        return None

    lines = [first]
    while True:
        text = "\n".join(lines)
        if text.count("```") % 2 == 1:
            try:
                lines.append(input())
                continue
            except EOFError:
                break
        if text.count('"') % 2 == 1:
            try:
                lines.append(input())
                continue
            except EOFError:
                break
        break
    return "\n".join(lines).strip()


def _extract_email_override(message: str) -> tuple[str, Optional[str]]:
    """
    Best-effort extraction of an email draft from the user's message.

    - If message contains a ``` fenced block, treat the first block as the email body.
    - Else, if message contains a long quoted block " ... ", treat it as the email body.
    - Else, no extraction.
    """
    if "```" in message:
        first = message.find("```")
        second = message.find("```", first + 3)
        if second != -1:
            body = message[first + 3 : second].strip()
            rest = (message[:first] + message[second + 3 :]).strip()
            if body:
                return rest, body

    if '"' in message:
        first = message.find('"')
        last = message.rfind('"')
        if last > first:
            body = message[first + 1 : last].strip()
            # Avoid treating short quoted snippets as an "email".
            if len(body) >= 120 and ("dear" in body.lower() or "\n" in body):
                rest = (message[:first] + message[last + 1 :]).strip()
                return rest, body

    return message, None


async def _maybe_build_real_llm(model: str) -> Optional[Any]:
    """
    Try to load llm-client + OpenAIProvider dynamically.
    Returns None if dependencies aren't available.
    """
    llm_src = os.path.join(repo_root, "llm-client", "src")
    if llm_src not in sys.path:
        sys.path.insert(0, llm_src)

    try:
        from llm_client import OpenAIProvider, load_env  # type: ignore
    except Exception:
        return None

    # Prefer loading repo-root .env (if present), then fall back to llm-client/.env.
    try:
        loaded = load_env(path=None, override=False)
    except Exception:
        loaded = False

    if not loaded:
        env_path = os.path.join(repo_root, "llm-client", ".env")
        try:
            load_env(path=env_path, override=False)
        except Exception:
            # If dotenv isn't available or env loading fails, we still try env vars.
            pass

    return OpenAIProvider(model=model)


async def main() -> None:
    parser = argparse.ArgumentParser(description="AI demo: switchboard + outreach email review/optimize")
    parser.add_argument("--real", action="store_true", help="Use llm-client OpenAIProvider (requires deps + API key)")
    parser.add_argument("--model", default="gpt-5-nano", help="Model key for llm-client (default: gpt-5-nano)")
    args = parser.parse_args()

    llm: Any
    if args.real:
        llm = await _maybe_build_real_llm(args.model)
        if llm is None:
            print("Could not import llm-client OpenAIProvider in this environment.")
            print("Run with mock mode (default), or use: llm-client/.venv/bin/python demo/run_ai_outreach_chat.py --real")
            return
    else:
        llm = MockLLM()

    kernel = AIDemoKernel(repo_root=repo_root, llm=llm)

    # Seed a sample platform state (this is what the real PlatformContextLoader would return).
    thread_id = 9001
    student_id = 88
    funding_request_id = 556
    kernel.seed_platform_context(
        thread_id=thread_id,
        student_id=student_id,
        funding_request_id=funding_request_id,
        platform_email_subject="Hello Prof. Demo",
        platform_email_body=(
            "Dear Prof. Demo,\n\n"
            "I’m reaching out to ask about opportunities in your lab. "
            "My interests include machine learning and systems.\n\n"
            "Best,\n"
            "Demo Student"
        ),
    )

    print(
        "Type a message (e.g. 'optimize my email') or 'exit'.\n\n"
        "Tip: for multi-line emails, wrap the email body in either:\n"
        "  - triple backticks ``` ... ```\n"
        "  - double quotes \" ... \"\n"
    )

    pending_action_id: Optional[str] = None
    pending_action_type: Optional[str] = None

    while True:
        user_message = _read_multiline_message()
        if user_message is None:
            break
        user_message = user_message.strip()
        if user_message.lower() in {"exit", "quit"}:
            break

        cleaned_message, email_override = _extract_email_override(user_message)

        intent_type, inputs = await kernel.switchboard_intent(cleaned_message)
        if email_override:
            inputs["email_text_override"] = email_override
        print(f"\n[switchboard] intent_type={intent_type} inputs={inputs}\n")

        run = await kernel.submit_intent(
            intent_type=intent_type,
            thread_id=thread_id,
            student_id=student_id,
            funding_request_id=funding_request_id,
            inputs=inputs,
            event_sink=_print_event,
        )

        await kernel.run_to_completion(run)

        pending_action_id = run.pending_action_id
        pending_action_type = run.pending_action_type

        # Handle a single waiting action inline (demo UX).
        if run.status == "waiting_action" and pending_action_id and pending_action_type:
            if pending_action_type == "collect_fields":
                print("\nPlease paste the email body to continue. End with an empty line.\n")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                body = "\n".join(lines).strip()
                payload = {"email_text_override": body} if body else {}
                await kernel.resolve_action_and_resume(action_id=pending_action_id, accepted=True, payload=payload)
                continue

            if pending_action_type == "apply_platform_patch":
                ans = input("\nApply the proposed changes? (y/n): ").strip().lower()
                await kernel.resolve_action_and_resume(
                    action_id=pending_action_id,
                    accepted=(ans == "y"),
                    payload={},
                )
                continue

        print("")  # spacer

    # Close real provider if used.
    close_fn = getattr(llm, "close", None)
    if callable(close_fn):
        try:
            await close_fn()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
