#!/usr/bin/env python3
"""
Example: Batch Processing with BatchManager

Demonstrates:
1. Processing many requests concurrently
2. Using checkpoints to resume interrupted batches
3. Progress tracking with tqdm
4. Error handling in batch operations
"""
import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    BatchManager,
    OpenAIClient,
)


async def main():
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create temporary directories
    cache_dir = Path(tempfile.mkdtemp(prefix="llm_cache_"))
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="llm_checkpoints_"))
    print(f"📁 Cache directory: {cache_dir}")
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # Initialize client
    client = OpenAIClient(
        model="gpt-5-nano",
        cache_backend="fs",
        cache_dir=cache_dir,
    )
    
    # === Example 1: Simple Batch Processing ===
    print("\n" + "=" * 40)
    print("Example 1: Simple Batch")
    print("=" * 40)
    
    # Create a batch of requests
    requests = [
        {"messages": [{"role": "user", "content": f"What is {i} + {i}? Reply with just the number."}]}
        for i in range(1, 6)
    ]
    
    print(f"\nProcessing {len(requests)} requests...")
    
    # Create batch manager
    manager = BatchManager(
        max_workers=3,  # Process 3 at a time
    )
    
    # Define the processor function
    async def processor(req):
        return await client.get_response(**req)
    
    # Process batch (no checkpoint)
    results = await manager.process_batch(requests, processor=processor)
    
    print("\nResults:")
    for i, result in enumerate(results):
        output = result.get("output", "ERROR")
        print(f"  {i+1} + {i+1} = {output}")
    
    # === Example 2: Batch with Checkpointing ===
    print("\n" + "=" * 40)
    print("Example 2: Batch with Checkpointing")
    print("=" * 40)
    
    # Larger batch with checkpoints
    requests = [
        {"messages": [{"role": "user", "content": f"Say the word 'test{i}'"}]}
        for i in range(10)
    ]
    
    checkpoint_file = checkpoint_dir / "batch_checkpoint.jsonl"
    
    manager = BatchManager(
        max_workers=3,
        checkpoint_file=checkpoint_file,
    )
    
    print(f"\nProcessing {len(requests)} requests with checkpoints...")
    
    results = await manager.process_batch(
        requests,
        processor=processor,
        desc="Example Batch",
    )
    
    print(f"\nCompleted {len(results)} requests")
    print(f"Sample result: {results[0].get('output', 'N/A')[:50]}...")
    
    # Show checkpoint file
    if checkpoint_file.exists():
        print(f"\nCheckpoint file created: {checkpoint_file}")
        print(f"Items in checkpoint: {len(checkpoint_file.read_text().strip().split('\n'))}")
    
    # === Example 3: Processing with Caching ===
    print("\n" + "=" * 40)
    print("Example 3: Batch with Caching")
    print("=" * 40)
    
    # Same requests - should hit cache
    import time
    
    requests = [
        {
            "messages": [{"role": "user", "content": f"Translate '{word}' to French"}],
            "cache_response": True,
            "cache_collection": "translations",
        }
        for word in ["hello", "world", "python", "code", "test"]
    ]
    
    manager = BatchManager(
        max_workers=5,
    )
    
    # First run - not cached
    start = time.perf_counter()
    results1 = await manager.process_batch(requests, processor=processor)
    time1 = time.perf_counter() - start
    print(f"\nFirst run: {len(results1)} results in {time1:.2f}s")
    
    # Second run - should hit cache
    start = time.perf_counter()
    results2 = await manager.process_batch(requests, processor=processor)
    time2 = time.perf_counter() - start
    print(f"Cached run: {len(results2)} results in {time2:.2f}s")
    print(f"Speedup: {time1/time2:.1f}x faster")
    
    # Show results
    print("\nTranslations:")
    words = ["hello", "world", "python", "code", "test"]
    for word, result in zip(words, results2):
        print(f"  {word} → {result.get('output', 'ERROR')[:30]}")
    
    # === Example 4: Error Handling ===
    print("\n" + "=" * 40)
    print("Example 4: Error Handling")
    print("=" * 40)
    
    # Mix of valid and problematic requests
    requests = [
        {"messages": [{"role": "user", "content": "Say 'success'"}]},
        {"messages": [{"role": "user", "content": "Say 'also success'"}]},
    ]
    
    manager = BatchManager(
        max_workers=2,
    )
    
    results = await manager.process_batch(requests, processor=processor)
    
    # Check results
    successes = sum(1 for r in results if r.get("status") == 200)
    errors = sum(1 for r in results if r.get("status") != 200)
    
    print(f"\nResults: {successes} successes, {errors} errors")
    
    for i, result in enumerate(results):
        status = result.get("status")
        if status == 200:
            print(f"  Request {i+1}: ✅ {result.get('output', '')[:30]}")
        else:
            print(f"  Request {i+1}: ❌ {result.get('error', 'Unknown error')}")
    
    # === Cleanup ===
    await client.close()
    
    # Show final cache stats
    print("\n" + "=" * 40)
    print("Cache Statistics")
    print("=" * 40)
    
    cache_files = list(cache_dir.rglob("*.json"))
    print(f"Total cached responses: {len(cache_files)}")
    
    for item in sorted(cache_dir.iterdir()):
        if item.is_dir():
            count = len(list(item.glob("*.json")))
            print(f"  📁 {item.name}/: {count} files")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
