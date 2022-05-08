import asyncio
import concurrent.futures
from typing import Any, Awaitable


def asyncio_run_in_thread(coro: Awaitable[Any]) -> Any:
    def _run() -> None:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_run)]):
            if future.exception():
                raise future.exception()  # type: ignore
            return future.result()
