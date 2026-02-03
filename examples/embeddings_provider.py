#!/usr/bin/env python3
"""
Example: Embeddings with OpenAIProvider

Demonstrates:
1. Generating embeddings with the new Provider API
2. Single and batch embedding generation
3. Similarity computation
4. Embedding dimensions and formats
"""

import asyncio
import math
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIProvider


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


async def main():
    print("=" * 60)
    print("EMBEDDINGS WITH PROVIDER API")
    print("=" * 60)

    # === Example 1: Basic Embedding ===
    print("\n" + "=" * 40)
    print("Example 1: Basic Embedding")
    print("=" * 40)

    # Use embedding model
    provider = OpenAIProvider(model="text-embedding-3-small")

    text = "The quick brown fox jumps over the lazy dog."

    result = await provider.embed(text)

    if result.ok:
        embedding = result.embedding  # Convenience for single input
        print(f"\nText: '{text}'")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Tokens used: {result.usage.input_tokens if result.usage else 'N/A'}")
    else:
        print(f"Error: {result.error}")

    await provider.close()

    # === Example 2: Batch Embeddings ===
    print("\n" + "=" * 40)
    print("Example 2: Batch Embeddings")
    print("=" * 40)

    provider = OpenAIProvider(model="text-embedding-3-small")

    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language.",
        "Cats are small domesticated mammals.",
    ]

    result = await provider.embed(texts)

    if result.ok:
        print(f"\nEmbedded {len(texts)} texts")
        print(f"Embeddings shape: {len(result.embeddings)} x {len(result.embeddings[0])}")
        print(f"Total tokens: {result.usage.total_tokens if result.usage else 'N/A'}")

        for i, text in enumerate(texts):
            print(f"  Text {i + 1}: '{text[:40]}...' → [{result.embeddings[i][0]:.4f}, ...]")

    await provider.close()

    # === Example 3: Semantic Similarity ===
    print("\n" + "=" * 40)
    print("Example 3: Semantic Similarity")
    print("=" * 40)

    provider = OpenAIProvider(model="text-embedding-3-small")

    # Similar pairs and different pairs
    sentences = [
        "The cat sat on the mat.",  # 0
        "A feline rested on the rug.",  # 1 (similar to 0)
        "Python is great for data science.",  # 2
        "Data analysis works well in Python.",  # 3 (similar to 2)
    ]

    result = await provider.embed(sentences)

    if result.ok:
        embeddings = result.embeddings

        print("\nSimilarity Matrix:")
        print("-" * 50)

        # Compute pairwise similarities
        similarities = []
        for i in range(len(sentences)):
            row = []
            for j in range(len(sentences)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                row.append(sim)
            similarities.append(row)

        # Print as table
        print("       ", end="")
        for i in range(len(sentences)):
            print(f"  S{i + 1}  ", end="")
        print()

        for i, row in enumerate(similarities):
            print(f"  S{i + 1}  ", end="")
            for sim in row:
                print(f" {sim:.3f}", end="")
            print()

        print("\nInterpretation:")
        print(f"  S1 ↔ S2 (cat sentences): {similarities[0][1]:.3f} (high - similar)")
        print(f"  S3 ↔ S4 (Python sentences): {similarities[2][3]:.3f} (high - similar)")
        print(f"  S1 ↔ S3 (cat vs Python): {similarities[0][2]:.3f} (lower - different)")

    await provider.close()

    # === Example 4: Reduced Dimensions ===
    print("\n" + "=" * 40)
    print("Example 4: Reduced Dimensions")
    print("=" * 40)

    provider = OpenAIProvider(model="text-embedding-3-small")

    text = "Embedding with reduced dimensions for efficiency."

    # Default dimensions
    result_full = await provider.embed(text)

    # Reduced dimensions (if model supports it)
    result_small = await provider.embed(text, dimensions=256)

    if result_full.ok and result_small.ok:
        print(f"\nFull dimensions: {len(result_full.embedding)}")
        print(f"Reduced dimensions: {len(result_small.embedding)}")

        # Reduced embeddings are normalized, can still compute similarity
        # (with some quality tradeoff for efficiency)

    await provider.close()

    # === Example 5: Encoding Formats ===
    print("\n" + "=" * 40)
    print("Example 5: Encoding Formats")
    print("=" * 40)

    provider = OpenAIProvider(model="text-embedding-3-small")

    text = "Testing different encoding formats."

    # Float format (directly usable)
    result_float = await provider.embed(text, encoding_format="float")

    # Base64 format (more efficient for transfer, auto-decoded)
    result_b64 = await provider.embed(text, encoding_format="base64")

    if result_float.ok and result_b64.ok:
        print(f"\nFloat format: {len(result_float.embedding)} values")
        print(f"Base64 format (decoded): {len(result_b64.embedding)} values")

        # They should be equivalent after decoding
        diff = sum(
            abs(a - b)
            for a, b in zip(
                result_float.embedding[:10],
                result_b64.embedding[:10],
                strict=False,
            )
        )
        print(f"Difference in first 10 values: {diff:.10f}")

    await provider.close()

    # === Example 6: Search Application ===
    print("\n" + "=" * 40)
    print("Example 6: Semantic Search")
    print("=" * 40)

    provider = OpenAIProvider(model="text-embedding-3-small")

    # Document corpus
    documents = [
        "Python is a high-level programming language.",
        "JavaScript runs in web browsers.",
        "Machine learning algorithms learn from data.",
        "Databases store and retrieve information.",
        "Neural networks are inspired by the brain.",
    ]

    # Get document embeddings
    doc_result = await provider.embed(documents)

    if doc_result.ok:
        doc_embeddings = doc_result.embeddings

        # Query
        query = "How do AI systems learn?"
        query_result = await provider.embed(query)

        if query_result.ok:
            query_embedding = query_result.embedding

            print(f"\nQuery: '{query}'")
            print("\nSearch results (by relevance):")

            # Compute similarities
            scores = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(query_embedding, doc_emb)
                scores.append((sim, documents[i]))

            # Sort by similarity
            for sim, doc in sorted(scores, reverse=True):
                print(f"  [{sim:.3f}] {doc}")

    await provider.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
