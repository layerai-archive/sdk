import logging
import uuid
from typing import Any, Callable, List

from layer.contracts.runs import Run
from layer.executables.ray_runtime import RayClientFunctionRuntime

from .base_project_runner import BaseProjectRunner


logger = logging.getLogger()


class RayProjectRunner(BaseProjectRunner):
    def __init__(self, functions: List[Any], ray_address: str) -> None:
        super().__init__(functions)
        self.ray_address = ray_address

    def _run(self, debug: bool = False, printer: Callable[[str], Any] = print) -> Run:
        for definition in self.definitions:
            definition.package()
            RayClientFunctionRuntime.execute(
                definition.executable_path,
                address=self.ray_address,
                fabric=definition.fabric,
            )

        run = Run(
            id=uuid.UUID(int=0),
        )  # TODO: Workflow integration with ray to obtain run id.
        return run
