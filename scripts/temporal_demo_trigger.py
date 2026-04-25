"""Kicks off a single DemoWorkflow run and waits for the result.

Run this in a third terminal (the worker is in another). Don't kill THIS
one — the whole point is to kill the worker, not the trigger. The trigger
just blocks waiting for the workflow to complete; the workflow itself is
running on the dev server's task queue.

Run:
    uv run python scripts/temporal_demo_trigger.py
"""

import asyncio
from uuid import uuid4

from rich import print
from temporalio.client import Client

from horcrux.config import settings


async def main() -> None:
    client = await Client.connect(
        settings.temporal.address,
        namespace=settings.temporal.namespace,
    )
    workflow_id = f"demo-{uuid4().hex[:8]}"
    handle = await client.start_workflow(
        "DemoWorkflow",
        id=workflow_id,
        task_queue="horcrux-demo",
    )
    print(f"[bold green]started[/] workflow_id={workflow_id}")
    print(f"watch in UI: [link]http://localhost:8233/namespaces/default/workflows/{workflow_id}[/link]")
    print("waiting for completion...")

    result = await handle.result()
    print(f"[bold green]completed[/] {len(result)} steps")
    for step in result:
        print(f"  · {step}")


if __name__ == "__main__":
    asyncio.run(main())
