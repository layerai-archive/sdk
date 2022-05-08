import asyncio
import signal
import sys
from signal import Signals
from typing import Any, Callable, Coroutine, Optional

from layer.exceptions import exception_handler


@exception_handler(stage="Post user script execution")  # pytype: disable=not-callable
def exit_code_handler(exit_code: int, err: Optional[str], logger: Any) -> None:
    logger.info(f"Process exit code: {exit_code}")
    if exit_code == 0:
        # 0 means successful termination or error has been reported with the
        # @exception_handler decorator, so do nothing here
        return
    elif exit_code > 128:
        # Signal terminated process
        signal_number = exit_code - 128
        signal_name = Signals(signal_number).name
        if sys.version_info[1] >= 8:
            # signal.strsignal since Python 3.8
            signal_description = signal.strsignal(signal_number)
            error_message = f"Process exit code: {exit_code}. Inferred interrupt signal {signal_name}({signal_number}): {signal_description}"
        else:
            error_message = f"Process exit code: {exit_code}. Inferred interrupt signal {signal_name}({signal_number})"
        raise Exception(error_message)
    else:
        extra_error_message = (
            f", error: {err}" if (err is not None and err != "") else ""
        )
        raise Exception(f"Process exit code: {exit_code}{extra_error_message}")


def run_in_async_event_loop(
    run_func: Coroutine[Any, Any, Any],
    error_reporting_callback: Callable[[str, Exception], None],
    logger: Any,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exit_code, err = loop.run_until_complete(run_func)
    exit_code_handler(exit_code, err, logger, callback=error_reporting_callback)
    exit(exit_code)
