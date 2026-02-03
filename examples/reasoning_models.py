#!/usr/bin/env python3
"""
Example: Reasoning Models

Demonstrates:
1. Using reasoning models with different effort levels
2. Comparing outputs across reasoning efforts
"""

import asyncio

# Add src to path for development
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient


async def main():
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_reasoning_cache_"))
    print(f"📁 Cache directory: {cache_dir}")

    # Use gpt-5-nano which supports reasoning
    client = OpenAIClient(
        model="gpt-5-nano",
        cache_backend="fs",
        cache_dir=cache_dir,
        cache_collection="reasoning_tests",
    )

    problem = """
    A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?
    Think step by step and explain your reasoning.
    """

    print("=" * 60)
    print("REASONING EFFORT COMPARISON")
    print("=" * 60)
    print(f"\n📝 Problem: {problem.strip()}\n")

    # Test different reasoning efforts
    efforts = ["minimal", "low", "medium", "high"]

    for effort in efforts:
        print("-" * 60)
        print(f"🧠 Reasoning Effort: {effort.upper()}")
        print("-" * 60)

        response = await client.get_response(
            messages=[{"role": "user", "content": problem}],
            reasoning_effort=effort,
            cache_response=True,
            cache_collection=f"reasoning_{effort}",
        )

        output = response.get("output", "")
        usage = response.get("usage", {})

        # Truncate output for display
        if len(output) > 500:
            display_output = output[:500] + "..."
        else:
            display_output = output

        print(f"\n{display_output}\n")
        print(
            f"📊 Tokens: input={usage.get('input_tokens', 0)}, "
            f"output={usage.get('output_tokens', 0)}, "
            f"cost=${usage.get('total_cost', 0):.6f}"
        )
        print()

    # --- Show cache structure ---
    print("=" * 60)
    print("CACHE STRUCTURE")
    print("=" * 60)

    for item in sorted(cache_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("*.json")))
            print(f"  📁 {item.name}/ ({file_count} files)")

    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
