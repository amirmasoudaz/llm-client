#!/usr/bin/env python3
"""
Example: Embeddings with Dynamic Collections

Demonstrates:
1. Embedding generation
2. Caching embeddings with dynamic collections
3. Similarity computation between embeddings
"""

import asyncio
import math

# Add src to path for development
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


async def main():
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_embed_cache_"))
    print(f"📁 Cache directory: {cache_dir}")

    # Initialize embeddings client
    client = OpenAIClient(
        model="text-embedding-3-small",
        cache_backend="fs",
        cache_dir=cache_dir,
        cache_collection="embeddings_default",
    )

    # --- Test 1: Generate embeddings ---
    print("\n" + "=" * 60)
    print("TEST 1: Generate Embeddings")
    print("=" * 60)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Python is a programming language.",
    ]

    embeddings = []
    for i, text in enumerate(texts):
        response = await client.get_response(
            input=text,
            cache_response=True,
            cache_collection="semantic_search",
        )
        embedding = response.get("output")
        embeddings.append(embedding)
        print(f"Text {i + 1}: '{text[:40]}...' -> {len(embedding)} dimensions")

    # --- Test 2: Similarity computation ---
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Similarity")
    print("=" * 60)

    sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
    sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])
    sim_1_2 = cosine_similarity(embeddings[1], embeddings[2])

    print(f"Similarity (fox sentence 1 vs fox sentence 2): {sim_0_1:.4f}")
    print(f"Similarity (fox sentence 1 vs python sentence): {sim_0_2:.4f}")
    print(f"Similarity (fox sentence 2 vs python sentence): {sim_1_2:.4f}")

    print("\n💡 Note: The two fox sentences should have higher similarity!")

    # --- Test 3: Cache verification ---
    print("\n" + "=" * 60)
    print("TEST 3: Cache Verification (re-fetch first text)")
    print("=" * 60)

    import time

    start = time.perf_counter()

    response_cached = await client.get_response(
        input=texts[0],
        cache_response=True,
        cache_collection="semantic_search",
    )

    elapsed = time.perf_counter() - start
    print(f"Fetched from cache in {elapsed * 1000:.2f}ms")
    print(f"Same embedding? {embeddings[0] == response_cached.get('output')}")

    # --- Show cache structure ---
    print("\n" + "=" * 60)
    print("CACHE STRUCTURE")
    print("=" * 60)

    for item in sorted(cache_dir.rglob("*")):
        if item.is_dir():
            rel_path = item.relative_to(cache_dir)
            print(f"  📁 {rel_path}/")

    file_count = len(list(cache_dir.rglob("*.json")))
    print(f"\n  Total cached files: {file_count}")

    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
