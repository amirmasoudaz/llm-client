#!/usr/bin/env python3
"""
Performance benchmarks for llm-client.

Run with: python benchmarks/run_benchmarks.py
"""

import sys
import time
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from statistics import mean, stdev

from llm_client.models import ModelProfile
from llm_client.perf import clear_fingerprint_cache, fingerprint, fingerprint_messages

# Import components to benchmark
from llm_client.serialization import (
    fast_json_dumps,
    fast_json_loads,
    stable_json_dumps,
)


def timeit(func, iterations: int = 1000):
    """Time a function and return stats."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": mean(times) * 1000,
        "stdev_ms": stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "iterations": iterations,
    }


def print_result(name: str, result: dict):
    """Pretty print benchmark result."""
    print(f"  {name}:")
    print(f"    Mean: {result['mean_ms']:.4f} ms")
    print(f"    Std:  {result['stdev_ms']:.4f} ms")
    print(f"    Min:  {result['min_ms']:.4f} ms  Max: {result['max_ms']:.4f} ms")
    print(f"    Iterations: {result['iterations']}")


# --- Test Data ---
SMALL_MESSAGE = {"role": "user", "content": "Hello, world!"}
MEDIUM_MESSAGE = {"role": "user", "content": "This is a medium-sized message with some content. " * 20}
LARGE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. " * 10},
    {"role": "user", "content": "Please help me with this task. " * 50},
    {"role": "assistant", "content": "I'd be happy to help! " * 50},
    {"role": "user", "content": "Thank you for your help! " * 50},
]
COMPLEX_DATA = {
    "messages": LARGE_MESSAGES,
    "model": "gpt-5-nano",
    "temperature": 0.7,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
                    "required": ["query"],
                },
            },
        }
    ],
}


def benchmark_serialization():
    """Benchmark JSON serialization."""
    print("\n=== Serialization Benchmarks ===")

    # Small object
    print("\nSmall object (single message):")
    print_result("stable_json_dumps", timeit(lambda: stable_json_dumps(SMALL_MESSAGE)))
    print_result("fast_json_dumps", timeit(lambda: fast_json_dumps(SMALL_MESSAGE)))
    print_result("json.dumps (baseline)", timeit(lambda: json.dumps(SMALL_MESSAGE)))

    # Large object
    print("\nLarge object (complex request):")
    print_result("stable_json_dumps", timeit(lambda: stable_json_dumps(COMPLEX_DATA), 500))
    print_result("fast_json_dumps", timeit(lambda: fast_json_dumps(COMPLEX_DATA), 500))
    print_result("json.dumps (baseline)", timeit(lambda: json.dumps(COMPLEX_DATA), 500))

    # Deserialization
    json_bytes = fast_json_dumps(COMPLEX_DATA)
    json_str = json.dumps(COMPLEX_DATA)
    print("\nDeserialization (complex request):")
    print_result("fast_json_loads (bytes)", timeit(lambda: fast_json_loads(json_bytes), 500))
    print_result("json.loads (string)", timeit(lambda: json.loads(json_str), 500))


def benchmark_fingerprinting():
    """Benchmark fingerprint operations."""
    print("\n=== Fingerprinting Benchmarks ===")

    clear_fingerprint_cache()

    # Cold fingerprinting
    print("\nCold fingerprinting (uncached):")
    clear_fingerprint_cache()
    print_result("fingerprint (small)", timeit(lambda: fingerprint(SMALL_MESSAGE)))

    clear_fingerprint_cache()
    print_result("fingerprint (complex)", timeit(lambda: fingerprint(COMPLEX_DATA), 500))

    # Message fingerprinting
    clear_fingerprint_cache()
    print_result("fingerprint_messages", timeit(lambda: fingerprint_messages(LARGE_MESSAGES)))

    # Cached fingerprinting (repeated calls)
    print("\nCached fingerprinting (repeated):")
    _ = fingerprint(COMPLEX_DATA)  # Prime cache
    print_result("fingerprint (cached)", timeit(lambda: fingerprint(COMPLEX_DATA), 500))


def benchmark_token_counting():
    """Benchmark token counting."""
    print("\n=== Token Counting Benchmarks ===")

    # Get a model profile
    profile = ModelProfile.get("gpt-5")

    # Clear any previous cache
    profile._count_tokens_str.cache_clear()

    short_text = "Hello, world!"
    medium_text = "This is a medium-sized piece of text. " * 50
    long_text = "This is a longer piece of text that simulates a real prompt. " * 200

    print("\nCold token counting (uncached):")
    profile._count_tokens_str.cache_clear()
    print_result("short text", timeit(lambda: profile.count_tokens(short_text)))

    profile._count_tokens_str.cache_clear()
    print_result("medium text", timeit(lambda: profile.count_tokens(medium_text), 500))

    profile._count_tokens_str.cache_clear()
    print_result("long text", timeit(lambda: profile.count_tokens(long_text), 100))

    # Cached
    print("\nCached token counting (repeated):")
    _ = profile.count_tokens(medium_text)  # Prime cache
    print_result("medium text (cached)", timeit(lambda: profile.count_tokens(medium_text), 500))

    # Message list
    print("\nMessage list token counting:")
    profile._count_tokens_str.cache_clear()
    print_result("message list", timeit(lambda: profile.count_tokens(LARGE_MESSAGES), 500))


def benchmark_cache_key_generation():
    """Benchmark cache key generation patterns."""
    print("\n=== Cache Key Generation Benchmarks ===")

    from blake3 import blake3

    from llm_client.serialization import stable_json_dumps

    # Simulate cache key generation
    def generate_cache_key():
        json_str = stable_json_dumps(COMPLEX_DATA)
        return blake3(json_str.encode()).hexdigest()

    clear_fingerprint_cache()

    print("\nFull cache key generation:")
    print_result("full generation", timeit(generate_cache_key, 500))

    print("\nUsing fingerprint utility:")
    clear_fingerprint_cache()
    print_result("fingerprint utility", timeit(lambda: fingerprint(COMPLEX_DATA), 500))


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("LLM-Client Performance Benchmarks")
    print("=" * 60)

    # Check for orjson
    import importlib.util

    if importlib.util.find_spec("orjson") is not None:
        print("\n✓ orjson is available - using fast path")
    else:
        print("\n⚠ orjson not installed - using standard json")
        print("  Install with: pip install llm-client[performance]")

    benchmark_serialization()
    benchmark_fingerprinting()
    benchmark_token_counting()
    benchmark_cache_key_generation()

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
