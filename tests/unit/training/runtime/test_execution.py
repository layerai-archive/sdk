import logging
import re
import signal
from typing import Optional, Tuple

import pytest

from layer.training.runtime.execution import exit_code_handler, run_in_async_event_loop


logger = logging.getLogger(__name__)


class CallbackClass:
    def __init__(self):
        self.callback_called = False
        self.stage = ""
        self.reason = ""

    def callback(self, stage: str, reason: Exception) -> None:
        self.callback_called = True
        self.stage = stage
        self.reason = reason


async def run_func(exit_code: int) -> Tuple[int, Optional[str]]:
    return exit_code, None


class TestExecution:
    def test_exit_code_handler_when_exit_code_success_or_reported_exception_do_nothing(
        self,
    ) -> None:
        callback_instance = CallbackClass()

        exit_code_handler(0, None, logger, callback=callback_instance.callback)

        assert not callback_instance.callback_called

    def test_exit_code_handler_when_exit_code_failure_then_report(self) -> None:
        callback_instance = CallbackClass()

        with pytest.raises(SystemExit):
            exit_code_handler(1, None, logger, callback=callback_instance.callback)

        assert callback_instance.callback_called
        assert callback_instance.stage == "Post user script execution"
        assert str(callback_instance.reason) == str(Exception("Process exit code: 1"))

    def test_run_in_async_event_loop_reports_signals(self) -> None:
        # given
        exit_code = 128 + signal.SIGSEGV
        # when
        callback_instance = self.__assert_exit_code(exit_code)
        # then
        expected_pattern = re.compile(
            f".*Process exit code: {exit_code}. Inferred interrupt signal SIGSEGV.*"
        )
        assert callback_instance.callback_called
        assert callback_instance.stage == "Post user script execution"
        assert expected_pattern.match(str(callback_instance.reason))

    def test_run_in_async_event_loop_reports_bad_exits(self) -> None:
        # given
        exit_code = 1
        # when
        callback_instance = self.__assert_exit_code(exit_code)
        # then
        expected_pattern = re.compile(".*Process exit code: 1.*")
        assert callback_instance.callback_called
        assert callback_instance.stage == "Post user script execution"
        assert expected_pattern.match(str(callback_instance.reason))

    def test_run_in_async_event_loop_does_not_report_on_good_exit(self) -> None:
        # given
        exit_code = 0
        # when
        callback_instance = self.__assert_exit_code(exit_code)
        # then
        assert not callback_instance.callback_called

    @staticmethod
    def __assert_exit_code(exit_code: int) -> CallbackClass:
        callback_instance = CallbackClass()
        with pytest.raises(SystemExit) as e:
            run_in_async_event_loop(
                run_func(exit_code), callback_instance.callback, logger
            )
            assert exit_code == e.value.code
        return callback_instance
