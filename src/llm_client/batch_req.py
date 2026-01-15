import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterable, List, Union

try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None


class BatchManager:
    """
    A robust manager for processing asynchronous batch jobs with:
    - Producer-Consumer worker pool (memory efficient)
    - Checkpointing (resume capability)
    - Error handling & Progress tracking
    """
    
    def __init__(
        self, 
        max_workers: int = 50,
        checkpoint_file: Union[str, Path, None] = None,
        save_interval: int = 10
    ):
        self.max_workers = max_workers
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
        self.save_interval = save_interval
        self.results = []
        self.processed_indices = set()
        
        # Load existing checkpoint if available
        if self.checkpoint_file and self.checkpoint_file.exists():
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Loads processed results from the checkpoint file."""
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    # Assuming some unique identifier or index is stored. 
                    # If the input doesn't have an ID, we might fallback to index if consistent.
                    # For simplicity, we will track by input index in the batch.
                    if "_batch_index" in record:
                        self.processed_indices.add(record["_batch_index"])
                        self.results.append(record)
            print(f"Loaded {len(self.processed_indices)} items from checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    async def process_batch(
        self, 
        items: Iterable[Any], 
        processor: Callable[[Any], Coroutine[Any, Any, Any]],
        desc: str = "Processing"
    ) -> List[Any]:
        """
        Process a batch of items using a worker pool.
        
        Args:
            items: Iterable of inputs to process.
            processor: Async function that takes an item and returns a result.
            desc: Description for the progress bar.
        
        Returns:
            List of results (including errors handled safely).
        """
        queue = asyncio.Queue()
        
        # Filter out already processed items
        items_to_process = []
        for i, item in enumerate(items):
            if i not in self.processed_indices:
                items_to_process.append((i, item))
        
        if not items_to_process:
            print("All items already processed in checkpoint.")
            # Return sorted results by index to maintain order
            return sorted(self.results, key=lambda x: x.get("_batch_index", -1))
        
        # Enqueue items
        for task_item in items_to_process:
            queue.put_nowait(task_item)
            
        # Create workers
        workers = []
        num_workers = min(self.max_workers, len(items_to_process))
        
        pbar = None
        if tqdm:
            pbar = tqdm(total=len(items_to_process), desc=desc)
            
        for _ in range(num_workers):
            worker = asyncio.create_task(self._worker(queue, processor, pbar))
            workers.append(worker)
            
        # Wait for queue to be empty
        await queue.join()
        
        # Cancel workers
        for w in workers:
            w.cancel()
        
        if pbar:
            pbar.close()
            
        # Return all results (new + loaded), sorted by index
        return sorted(self.results, key=lambda x: x.get("_batch_index", -1))

    async def _worker(self, queue: asyncio.Queue, processor: Callable, pbar):
        while True:
            index, item = None, None
            try:
                index, item = await queue.get()
                
                try:
                    result = await processor(item)
                    # If result is strict type (int/str), wrap it. 
                    # If dict, inject index.
                    if isinstance(result, dict):
                        out_record = result
                    else:
                        out_record = {"result": result}
                        
                except Exception as e:
                    out_record = {"error": str(e), "status": 500}

                out_record["_batch_index"] = index
                self.results.append(out_record)
                
                if self.checkpoint_file:
                    self._append_checkpoint(out_record)
                
                if pbar:
                    pbar.update(1)
                
                queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker critical error: {e}")
                # If we got an item but failed, we still need to mark it done
                if index is not None:
                    queue.task_done()

    def _append_checkpoint(self, record: dict):
        """Appends a single result to the checkpoint file."""
        try:
            with open(self.checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Failed to write checkpoint: {e}")

# Backward compatibility alias
class RequestManager(BatchManager):
    def __init__(self, max_semaphore: int = 1000):
        super().__init__(max_workers=max_semaphore)

    async def run_batch(self, coros: List[Coroutine] = None):
        """Adapter for legacy run_batch using list of coroutines."""
        if coros is None:
            coros = []
        
        # In legacy mode, we don't have inputs separate from coroutines.
        # We just await them. Checkpointing is harder here without serializable inputs.
        # We will just run them.
        
        async def _wrapper(coro):
            return await coro
            
        results = await self.process_batch(coros, _wrapper, desc="Legacy Batch")
        # Unwrap results to match legacy expectations
        final = []
        for r in results:
            if "error" in r and r.get("status") == 500:
                # Legacy behavior returned the dict with error
                final.append(r)
            else:
                # Legacy behavior returned the raw result
                final.append(r.get("result"))
        return final

__all__ = ["BatchManager", "RequestManager"]
