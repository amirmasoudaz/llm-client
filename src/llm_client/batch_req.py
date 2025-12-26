import asyncio
from typing import Coroutine, List


class RequestManager:
    DEFAULT_MAX_SEMAPHORE = 1000

    def __init__(self, max_semaphore: int = DEFAULT_MAX_SEMAPHORE):
        self.semaphore = asyncio.Semaphore(max_semaphore)
        self.tasks: List[Coroutine] = []
        self.results: List[dict] = []

    def add_task(self, task: Coroutine):
        self.tasks.append(task)

    async def run_one(self, task: Coroutine):
        async with self.semaphore:
            return await task

    async def run_batch(self, coros: List[Coroutine] = None):
        if coros is None:
            coros = self.tasks

        wrapped = [self.run_one(c) for c in coros]
        results = await asyncio.gather(*wrapped, return_exceptions=True)

        out = []
        for result in results:
            if isinstance(result, Exception):
                out.append({"error": str(result), "status": 500, "output": None, "usage": None})
            else:
                out.append(result)
        return out

    async def iter_batch(self, coros: list):
        for fut in asyncio.as_completed([self.run_one(c) for c in coros]):
            yield await fut


__all__ = ["RequestManager"]
