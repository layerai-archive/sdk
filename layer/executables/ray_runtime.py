from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import ray  # type: ignore # pylint: disable=import-error
from ray.client_builder import (  # type:ignore #  pylint: disable=import-error
    ClientContext,
)

from layer.config.config_manager import ConfigManager
from layer.contracts.fabrics import Fabric
from layer.executables.entrypoint_layer_runtime import EntrypointLayerFunctionRuntime
from layer.executables.packager import FunctionPackageInfo
from layer.executables.runtime import BaseFunctionRuntime
from layer.global_context import current_project_full_name


@ray.remote
def ray_runtime(executable_path: Path) -> None:
    EntrypointLayerFunctionRuntime.execute(executable_path=Path(executable_path.name))


class RayClientFunctionRuntime(BaseFunctionRuntime):
    def __init__(
        self, executable_path: Path, *args: Any, **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(executable_path)
        self._address = kwargs["address"]
        self._config = ConfigManager().load()
        self._client: Optional[ClientContext] = None
        self._fabric: Optional[Fabric] = None

    def initialise(self, package_info: FunctionPackageInfo) -> None:
        if not self._address:
            raise ValueError("Ray address is required!")
        project = current_project_full_name()
        if project is None:
            raise ValueError("project not specified and could not be resolved")

        runtime_env = {
            "working_dir": f"{self.executable_path.parent}",
            "env_vars": {
                "LAYER_PROJECT_NAME": project.project_name,
                "LAYER_CLIENT_AUTH_URL": str(self._config.url),
                "LAYER_CLIENT_AUTH_TOKEN": self._config.client.access_token,
            },
        }
        if package_info.conda_env:
            environment = package_info.conda_env.environment
            if "name" not in environment:
                environment["name"] = "layer"
            dependencies = environment["dependencies"]
            if "pip" not in dependencies:
                dependencies.append("pip")
                dependencies.append({"pip": ["layer"]})
                environment["dependencies"] = dependencies
            else:
                pip_dict: Dict[str, Any] = next(
                    (
                        item
                        for item in dependencies
                        if isinstance(item, dict) and "pip" in item
                    ),
                    {},
                )
                if pip_dict:
                    pip_dict["pip"].append("layer")
                else:
                    pip_dict["pip"] = ["layer"]
            environment["dependencies"] = "layer"
            runtime_env["conda"] = environment
        else:
            runtime_env["pip"] = ["layer", *[p for p in package_info.pip_dependencies]]

        self._fabric: Fabric = Fabric(  # type:ignore # pylint: disable=E1120;
            package_info.metadata["function"]["fabric"]["name"]
        )
        print(f"Connecting to the Ray instance at {self._address}")

        self._client = ray.init(
            address=self._address,
            runtime_env=runtime_env,
            log_to_driver=True,
            namespace="layer",
        )

    @property
    def executable_path(self) -> Path:
        return self._executable_path

    def install_packages(self, packages: Sequence[str]) -> None:
        pass

    def run_executable(self) -> Any:
        if self._client is None:
            raise ValueError("No client connection to Ray")
        with self._client:
            if self._fabric is None:
                raise ValueError("fabric is not specified and could not be resolved")
            ids = ray_runtime.options(
                num_cpus=self._fabric.cpu, num_gpus=self._fabric.gpu
            ).remote(self.executable_path)
            ray.wait([ids], num_returns=1)
        self._client.disconnect()
