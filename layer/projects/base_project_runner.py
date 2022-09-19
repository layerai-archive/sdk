import abc
from typing import Any, Callable, List

from layer.contracts.definitions import FunctionDefinition
from layer.contracts.runs import Run
from layer.runs import context


class BaseProjectRunner(abc.ABC):
    def __init__(self, functions: List[Any]) -> None:
        self.definitions: List[FunctionDefinition] = [
            f.get_definition_with_bound_arguments() for f in functions
        ]

    def run(self, debug: bool = False, printer: Callable[[str], Any] = print) -> Run:
        try:
            return self._run()
        except Exception as e:
            context.set_error(e)
            raise e

    @abc.abstractmethod
    def _run(self, debug: bool = False, printer: Callable[[str], Any] = print) -> Run:
        ...
