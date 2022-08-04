import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, List

from layerapi.api.ids_pb2 import RunId

from layer.config.config import DEFAULT_FUNC_PATH, Config
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.executables.function import Function
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
        self.functions: List[Function] = [Function.from_decorated(f) for f in functions]
        self.ray_address = ray_address

    def run(self) -> Run:
        for function in self.functions:
            asset_path = AssetPath(
                asset_type=AssetType(function.output_type_name),
                asset_name=function.output.name,
                project_name=self.project_full_name.project_name,
                org_name=self.project_full_name.account_name,
            )
            output_dir = DEFAULT_FUNC_PATH / asset_path.path()
            self._clean_output_dir(output_dir)
            executable_package_path = function.package(output_dir=output_dir)
            RayClientFunctionRuntime.execute(
                executable_package_path, address=self.ray_address
            )

        run = Run(
            id=RunId(value=str(uuid.UUID(int=0))),
            project_full_name=self.project_full_name,
        )  # TODO: Workflow integration with ray to obtain run id.
        return run

    def _clean_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
