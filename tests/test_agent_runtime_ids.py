from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_job_run_id_can_be_overridden():
    from agent_runtime.jobs.manager import JobManager, JobSpec
    from agent_runtime.jobs.store import InMemoryJobStore

    jm = JobManager(store=InMemoryJobStore())
    job = await jm.start(JobSpec(scope_id="t1", run_id="workflow-123"))
    assert job.run_id == "workflow-123"


@pytest.mark.asyncio
async def test_execution_context_has_per_job_cancellation_token():
    from agent_runtime.jobs.manager import JobManager, JobSpec
    from agent_runtime.jobs.store import InMemoryJobStore

    jm = JobManager(store=InMemoryJobStore())
    job1 = await jm.start(JobSpec(scope_id="t1"))
    job2 = await jm.start(JobSpec(scope_id="t1"))

    ctx1 = jm.create_context(job1)
    ctx2 = jm.create_context(job2)

    # When llm-client is available, these should be distinct token objects per job.
    if ctx1.cancel is not None and ctx2.cancel is not None:
        assert ctx1.cancel is not ctx2.cancel

