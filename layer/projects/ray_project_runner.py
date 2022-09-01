import logging
import uuid
from typing import Any, List

from layerapi.api.ids_pb2 import RunId

from layer.config.config import Config
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.executables.ray_runtime import RayClientFunctionRuntime


logger = logging.getLogger()


class RayProjectRunner:
    def __init__(
        self,
        config: Config,
        project_full_name: ProjectFullName,
        functions: List[Any],
        ray_address: str,
    ) -> None:
        self._config = config
        self.project_full_name = project_full_name
        self.definitions: List[FunctionDefinition] = [
            f.get_definition_with_bound_arguments() for f in functions
        ]
        self.ray_address = ray_address

    def run(self) -> Run:
        for definition in self.definitions:
            definition.package()
            RayClientFunctionRuntime.execute(
                definition.executable_path,
                address=self.ray_address,
                fabric=definition.fabric,
            )

        run = Run(
            id=RunId(value=str(uuid.UUID(int=0))),
            project_full_name=self.project_full_name,
        )  # TODO: Workflow integration with ray to obtain run id.
        return run
