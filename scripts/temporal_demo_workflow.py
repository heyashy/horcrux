"""Pure workflow + activity definitions for the demo crash test.

This module is *deliberately* minimal in its imports — only stdlib and
temporalio. It must be safely re-importable inside the Temporal worker
sandbox, which trips on libraries with circular or dynamic imports
(notably `rich`).

If you ever see `RuntimeError: Failed validating workflow ...` from the
Worker init, look for non-stdlib imports at the top of this file.
"""

import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# ── Activity ──────────────────────────────────────────────────────
# Activities are where side-effectful, non-deterministic work lives.
# They CAN use the wall clock, random, file I/O. Their results are
# persisted to event history when they complete.

@activity.defn
async def slow_step(step_num: int) -> str:
    activity.logger.info(f"step {step_num}: starting")
    await asyncio.sleep(3)
    activity.logger.info(f"step {step_num}: done")
    return f"step-{step_num}-result"


# ── Workflow ──────────────────────────────────────────────────────
# Workflow code MUST be deterministic. No datetime.now(), no random, no
# file I/O, no asyncio.sleep — use workflow.now() / workflow.uuid4() /
# workflow.sleep() instead. On worker restart, this code is REPLAYED from
# event history; non-determinism would break replay.

@workflow.defn
class DemoWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        results: list[str] = []
        for i in range(1, 11):
            result = await workflow.execute_activity(
                slow_step,
                args=[i],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            results.append(result)
        return results
