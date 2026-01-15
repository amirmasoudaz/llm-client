#!/usr/bin/env python3
"""
Example: Rate Limiting

Demonstrates:
1. Token bucket rate limiting
2. Using Limiter with providers
3. Manual token bucket usage
4. Understanding rate limit behavior
"""
import asyncio
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    OpenAIProvider,
    Limiter,
    TokenBucket,
    GPT5Nano,
)


async def main():
    print("=" * 60)
    print("RATE LIMITING EXAMPLE")
    print("=" * 60)
    
    # === Example 1: Automatic Rate Limiting ===
    print("\n" + "=" * 40)
    print("Example 1: Automatic Rate Limiting")
    print("=" * 40)
    
    print("\nOpenAIProvider includes automatic rate limiting based on model limits.")
    print(f"GPT5Nano limits: {GPT5Nano.request_limit} req/min, {GPT5Nano.token_limit} tokens/min")
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    # Make a few requests - rate limiter handles pacing automatically
    print("\nMaking 3 requests with automatic rate limiting...")
    
    for i in range(3):
        start = time.perf_counter()
        result = await provider.complete(f"Say 'request {i+1}'")
        elapsed = time.perf_counter() - start
        print(f"  Request {i+1}: {result.content} ({elapsed*1000:.0f}ms)")
    
    await provider.close()
    
    # === Example 2: Token Bucket Basics ===
    print("\n" + "=" * 40)
    print("Example 2: Token Bucket Basics")
    print("=" * 40)
    
    # Create a small bucket for demonstration
    bucket = TokenBucket(
        capacity=100,      # Maximum tokens the bucket can hold
        refill_rate=10,    # Tokens added per second
    )
    
    print(f"\nBucket: capacity=100, refill_rate=10/s")
    print(f"Initial tokens: {bucket.available}")
    
    # Acquire some tokens
    await bucket.acquire(50)
    print(f"After acquiring 50: {bucket.available} available")
    
    await bucket.acquire(30)
    print(f"After acquiring 30: {bucket.available} available")
    
    # Wait for refill
    print("\nWaiting 2 seconds for refill...")
    await asyncio.sleep(2)
    print(f"After 2s refill: {bucket.available} available (gained ~20 tokens)")
    
    # Try to acquire more than available (will wait)
    print("\nTrying to acquire 50 tokens (may need to wait)...")
    start = time.perf_counter()
    await bucket.acquire(50)
    elapsed = time.perf_counter() - start
    print(f"Acquired 50 tokens (waited {elapsed:.2f}s for refill)")
    
    # === Example 3: Limiter with Context Manager ===
    print("\n" + "=" * 40)
    print("Example 3: Limiter Context Manager")
    print("=" * 40)
    
    # Create a limiter based on model limits
    limiter = Limiter(GPT5Nano)
    
    print(f"\nLimiter for GPT5Nano:")
    print(f"  Request limit: {limiter.request_bucket.capacity}/min")
    print(f"  Token limit: {limiter.token_bucket.capacity}/min")
    
    # Use limiter context manager
    async with limiter.limit(tokens=100, requests=1) as ctx:
        print("\n  Inside rate limit context...")
        print(f"  Reserved 100 input tokens, 1 request")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        # Update with actual output tokens (for accurate tracking)
        ctx.output_tokens = 25
        print(f"  Reporting 25 output tokens")
    
    print("  Context exited - tokens released")
    
    # === Example 4: Concurrent Requests with Rate Limiting ===
    print("\n" + "=" * 40)
    print("Example 4: Concurrent Rate-Limited Requests")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    async def make_request(n: int):
        start = time.perf_counter()
        result = await provider.complete(f"Say 'concurrent {n}'")
        elapsed = time.perf_counter() - start
        return n, result.content, elapsed
    
    print("\nMaking 5 concurrent requests...")
    print("(Rate limiter will pace them automatically)")
    
    start_all = time.perf_counter()
    
    # Launch all requests concurrently
    tasks = [make_request(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_all
    
    # Show results in completion order
    for n, content, elapsed in sorted(results, key=lambda x: x[2]):
        print(f"  Request {n}: {content} ({elapsed*1000:.0f}ms)")
    
    print(f"\nTotal time for 5 requests: {total_time:.2f}s")
    
    await provider.close()
    
    # === Example 5: Manual Rate Control ===
    print("\n" + "=" * 40)
    print("Example 5: Manual Rate Control")
    print("=" * 40)
    
    # Create a very restrictive bucket for demo
    bucket = TokenBucket(
        capacity=50,       # Only 50 tokens
        refill_rate=100,   # Fast refill for demo
    )
    
    print("\nDemonstrating token acquisition timing...")
    
    for i in range(5):
        tokens_needed = 20
        
        start = time.perf_counter()
        await bucket.acquire(tokens_needed)
        wait_time = time.perf_counter() - start
        
        print(f"  Batch {i+1}: acquired {tokens_needed} tokens "
              f"(waited {wait_time*1000:.0f}ms, {bucket.available:.0f} remaining)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
