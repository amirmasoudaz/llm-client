#!/usr/bin/env python3
"""
Example: JSON Mode and Structured Output

Demonstrates:
1. JSON object response format
2. Pydantic structured output parsing
3. Caching structured responses
"""

import asyncio

# Add src to path for development
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient


class MovieReview(BaseModel):
    """Structured movie review schema."""

    title: str
    rating: float
    summary: str
    pros: list[str]
    cons: list[str]


async def main():
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_json_cache_"))
    print(f"📁 Cache directory: {cache_dir}")

    client = OpenAIClient(
        model="gpt-5-nano",
        cache_backend="fs",
        cache_dir=cache_dir,
        cache_collection="structured_outputs",
    )

    # --- Test 1: JSON Object Mode ---
    print("\n" + "=" * 60)
    print("TEST 1: JSON Object Mode")
    print("=" * 60)

    response = await client.get_response(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
            {
                "role": "user",
                "content": "List 3 programming languages with their primary use case. "
                "Return as JSON with keys: languages (array of {name, use_case}).",
            },
        ],
        response_format="json_object",
        cache_response=True,
        cache_collection="json_responses",
    )

    print(f"Status: {response.get('status')}")
    print(f"Output type: {type(response.get('output'))}")
    print(f"Output: {response.get('output')}")

    # --- Test 2: Pydantic Structured Output ---
    print("\n" + "=" * 60)
    print("TEST 2: Pydantic Structured Output")
    print("=" * 60)

    response = await client.get_response(
        messages=[
            {"role": "system", "content": "You are a movie critic. Provide structured reviews."},
            {"role": "user", "content": "Write a brief review of the movie 'Inception' (2010)."},
        ],
        response_format=MovieReview,
        cache_response=True,
        cache_collection="movie_reviews",
    )

    print(f"Status: {response.get('status')}")
    output = response.get("output")
    if isinstance(output, dict):
        print(f"Title: {output.get('title')}")
        print(f"Rating: {output.get('rating')}/10")
        print(f"Summary: {output.get('summary')}")
        print(f"Pros: {output.get('pros')}")
        print(f"Cons: {output.get('cons')}")

    # --- Test 3: Cache hit with structured output ---
    print("\n" + "=" * 60)
    print("TEST 3: Cache Hit with Structured Output")
    print("=" * 60)

    import time

    start = time.perf_counter()

    response_cached = await client.get_response(
        messages=[
            {"role": "system", "content": "You are a movie critic. Provide structured reviews."},
            {"role": "user", "content": "Write a brief review of the movie 'Inception' (2010)."},
        ],
        response_format=MovieReview,
        cache_response=True,
        cache_collection="movie_reviews",
    )

    elapsed = time.perf_counter() - start
    print(f"Fetched from cache in {elapsed * 1000:.2f}ms")
    print(f"Same output? {output == response_cached.get('output')}")

    # --- Show cache structure ---
    print("\n" + "=" * 60)
    print("CACHE COLLECTIONS")
    print("=" * 60)

    for item in sorted(cache_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("*.json")))
            print(f"  📁 {item.name}/ ({file_count} files)")

    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
