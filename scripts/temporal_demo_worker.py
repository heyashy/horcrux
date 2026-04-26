"""Demo worker — kill this with Ctrl+C to trigger the crash test.

The workflow + activity definitions live in `temporal_demo_workflow.py`.
This file deliberately does NOT import them at module top — it imports
inside main() so the Rich-based logging setup runs first and the workflow
module's clean re-imports during sandbox validation aren't tangled with
ours.

(In practice, even top-level import is safe as long as the *workflow file
itself* has only clean imports. But importing-inside-main is the bullet-
proof pattern.)

Run:
    uv run python scripts/temporal_demo_worker.py
"""

import asyncio
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("temporal-demo")


async def main() -> None:
    # Import inside main so any sandbox-sensitive setup happens first.
    from temporal_demo_workflow import DemoWorkflow, slow_step
    from temporalio.client import Client
    from temporalio.worker import Worker

    from horcrux.config import settings

    client = await Client.connect(
        settings.temporal.address,
        namespace=settings.temporal.namespace,
    )
    worker = Worker(
        client,
        task_queue="horcrux-demo",
        workflows=[DemoWorkflow],
        activities=[slow_step],
    )
    log.info(
        "[bold]worker started[/] task_queue=horcrux-demo · "
        "kill with Ctrl+C to test the crash recovery",
        extra={"markup": True},
    )
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("[yellow]worker stopped — restart me to see resume[/]", extra={"markup": True})
