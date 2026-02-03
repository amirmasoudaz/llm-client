#!/usr/bin/env python3
"""
Professor Reply Classification Agent

Analyzes professor email replies to extract feedback or criticism about
the outreach mechanism/approach used by the student.

Features:
- GPT-5-nano for cost-effective processing
- Response caching to avoid reprocessing
- Structured JSON output using Pydantic
- Batch processing with progress tracking
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pydantic import BaseModel, Field

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient



class OutreachFeedback(BaseModel):
    """Schema for extracting feedback about outreach approach from professor replies."""

    has_feedback: bool = Field(
        description="Whether the professor provided any feedback, praise, or criticism "
        "about HOW the student contacted them (their outreach approach/mechanism), "
        "not just about their qualifications or lab capacity."
    )
    feedback_type: Literal["praise", "criticism", "suggestion", "neutral", "none"] = Field(
        description="Type of feedback: 'praise' for positive comments about outreach, "
        "'criticism' for negative comments, 'suggestion' for advice on how to improve, "
        "'neutral' for matter-of-fact observations, 'none' if no outreach feedback."
    )
    feedback_quote: str = Field(
        description="Direct quote from the email containing the feedback about outreach. "
        "Empty string if no feedback found."
    )
    feedback_summary: str = Field(
        description="Brief summary of what the professor said about the outreach approach. "
        "Empty string if no feedback found."
    )
    confidence: float = Field(
        description="Confidence score 0.0-1.0 that this is genuine feedback about outreach "
        "mechanism rather than just standard acceptance/rejection language.", ge=0.0, le=1.0
    )


SYSTEM_PROMPT = """You are an expert at analyzing email communications. Your task is to identify when professors provide SUBSTANTIVE feedback, praise, or criticism specifically about HOW a prospective student contacted them (their outreach strategy, email quality, or approach).

IMPORTANT: Be VERY strict. Generic politeness is NOT feedback about outreach.

DEFINITELY NOT outreach feedback (set has_feedback=False):
- "Thanks for reaching out" - this is just politeness
- "Thank you for your interest" - standard acknowledgment
- "Sorry for the late reply" - not about outreach quality
- "My lab is full" / "not taking students" - capacity, not outreach
- "You're not the right fit" - about qualifications, not outreach
- Any automated/template response

ACTUALLY IS outreach feedback (set has_feedback=True):
- "You are doing a better job at this than most contacting me" - COMPARATIVE praise of approach
- "Your personalized email referencing my paper stood out" - praising customization
- "Please don't send mass/generic emails" - criticism of approach
- "Using ChatGPT to write emails is obvious" - criticism of approach  
- "A personal referral would be more effective" - ADVICE on strategy
- "Your persistence/follow-ups impressed me" - commenting on strategy
- "Your email was too long/unprofessional" - criticism of email quality
- Explicit advice about how to contact professors better

The key distinguisher: Does the professor specifically COMMENT ON or EVALUATE the student's outreach METHOD/STRATEGY, or are they just being polite while rejecting?

Set confidence HIGH (0.8+) only if there's explicit commentary on outreach approach.
Set confidence LOW (<0.3) for anything that could be generic politeness."""

USER_PROMPT_TEMPLATE = """Analyze this professor's reply email and extract any feedback about the student's outreach approach:

---EMAIL START---
{email_body}
---EMAIL END---

Remember: Only extract feedback specifically about HOW the student contacted them (the outreach mechanism), not about their qualifications or lab availability."""


async def classify_reply(client: OpenAIClient, email_body: str, funding_request_id: int) -> dict:
    """Classify a single professor reply."""
    try:
        response = await client.get_response(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(email_body=email_body)},
            ],
            response_format=OutreachFeedback,
            cache_response=True,
            cache_collection="professor_classification_v2",
        )

        output = response.get("output", {})
        # Handle case where output is a JSON string
        if isinstance(output, str):
            output = json.loads(output)
        
        return {
            "funding_request_id": funding_request_id,
            "status": "success",
            "has_feedback": output.get("has_feedback", False),
            "feedback_type": output.get("feedback_type", "none"),
            "feedback_quote": output.get("feedback_quote", ""),
            "feedback_summary": output.get("feedback_summary", ""),
            "confidence": output.get("confidence", 0.0),
            "email_preview": email_body[:200] + "..." if len(email_body) > 200 else email_body,
        }
    except Exception as e:
        return {
            "funding_request_id": funding_request_id,
            "status": "error",
            "error": str(e),
            "has_feedback": False,
            "feedback_type": "none",
            "feedback_quote": "",
            "feedback_summary": "",
            "confidence": 0.0,
            "email_preview": email_body[:200] + "..." if len(email_body) > 200 else email_body,
        }


async def process_all_replies(
    client: OpenAIClient,
    replies: list[dict],
    max_workers: int = 10,
    limit: int | None = None,
) -> list[dict]:
    """Process all replies with custom progress tracking."""
    if limit:
        replies = replies[:limit]

    total = len(replies)
    print(f"\n📧 Processing {total} professor replies...")

    # Progress tracking counters
    counters = {"done": 0, "failed": 0}
    results: list[dict] = []
    lock = asyncio.Lock()

    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # Create progress bar with standard format
    pbar = None
    if tqdm:
        pbar = tqdm(total=total, desc="✅0 ❌0 📋" + str(total), unit="email")

    async def process_one(email_body: str, funding_request_id: int) -> dict:
        """Process a single email and update counters."""
        result = await classify_reply(client, email_body, funding_request_id)

        async with lock:
            results.append(result)
            if result.get("status") == "success":
                counters["done"] += 1
            else:
                counters["failed"] += 1

            if pbar:
                remaining = total - counters["done"] - counters["failed"]
                pbar.set_description(f"✅{counters['done']} ❌{counters['failed']} 📋{remaining}")
                pbar.update(1)

        return result

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def bounded_process(email_body: str, funding_request_id: int) -> dict:
        async with semaphore:
            return await process_one(email_body, funding_request_id)

    # Create all tasks
    tasks = [
        bounded_process(r["professor_reply_body"], r["funding_request_id"])
        for r in replies
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    # Sort by index to maintain order
    results.sort(key=lambda x: x.get("funding_request_id", 0))

    return results


def display_results(results: list[dict]):
    """Display the results in a formatted way."""
    # Filter to emails with outreach feedback
    with_feedback = [r for r in results if r.get("has_feedback") and r.get("confidence", 0) >= 0.5]

    print("\n" + "=" * 80)
    print("📊 CLASSIFICATION RESULTS")
    print("=" * 80)

    print(f"\n📨 Total emails processed: {len(results)}")
    print(f"✅ Emails with outreach feedback: {len(with_feedback)}")
    print(f"📉 Success rate: {sum(1 for r in results if r['status'] == 'success') / len(results) * 100:.1f}%")

    # Count by feedback type
    type_counts = {}
    for r in with_feedback:
        ftype = r.get("feedback_type", "none")
        type_counts[ftype] = type_counts.get(ftype, 0) + 1

    if type_counts:
        print("\n📊 Feedback types:")
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            emoji = {"praise": "👍", "criticism": "⚠️", "suggestion": "💡", "neutral": "➖"}.get(ftype, "❓")
            print(f"   {emoji} {ftype}: {count}")

    # Show top findings by confidence
    if with_feedback:
        print("\n" + "=" * 80)
        print("🔍 TOP OUTREACH FEEDBACK FINDINGS")
        print("=" * 80)

        sorted_feedback = sorted(with_feedback, key=lambda x: -x.get("confidence", 0))

        for i, result in enumerate(sorted_feedback[:20], 1):  # Show top 20
            print(f"\n{'─' * 80}")
            print(f"📧 #{i} | Index: {result['funding_request_id']} | Type: {result['feedback_type'].upper()} | Confidence: {result['confidence']:.0%}")
            print(f"{'─' * 80}")

            if result.get("feedback_quote"):
                print(f"\n💬 Quote: \"{result['feedback_quote']}\"")

            if result.get("feedback_summary"):
                print(f"\n📝 Summary: {result['feedback_summary']}")

            print(f"\n📄 Email preview: {result['email_preview'][:300]}...")

    # Save detailed results to file
    output_file = Path(__file__).parent / "classification_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "total_processed": len(results),
                "with_feedback_count": len(with_feedback),
                "feedback_by_type": type_counts,
                "all_results": results,
                "with_feedback": sorted(with_feedback, key=lambda x: -x.get("confidence", 0)),
            },
            f,
            indent=2,
        )
    print(f"\n💾 Full results saved to: {output_file}")


async def main():
    """Main entry point."""
    print("=" * 80)
    print("🎓 PROFESSOR REPLY CLASSIFICATION AGENT")
    print("=" * 80)

    # Setup paths
    data_file = Path(__file__).parent / "professor_replies.json"
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"\n📁 Data file: {data_file}")
    print(f"📁 Cache directory: {cache_dir}")

    # Load replies
    with open(data_file) as f:
        replies = json.load(f)

    print(f"📊 Loaded {len(replies)} professor replies")

    # Initialize client with caching
    client = OpenAIClient(
        model="gpt-5-nano",
        cache_backend="fs",
        cache_dir=cache_dir,
        cache_collection="professor_classification",
    )

    try:
        # Process all replies (set limit=100 for testing, None for full dataset)
        results = await process_all_replies(
            client,
            replies,
            max_workers=15,  # Parallel requests
            limit=None,  # Set to e.g. 100 for testing
        )

        # Display and save results
        display_results(results)

    finally:
        await client.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
