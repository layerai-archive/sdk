from argparse import ArgumentError
from pathlib import Path
from typing import Any, Sequence

import ray

from layer.config.config_manager import ConfigManager
from layer.contracts.fabrics import Fabric
from layer.executables.packager import FunctionPackageInfo
from layer.executables.runtime.layer_runtime import LayerFunctionRuntime
from layer.executables.runtime.runtime import BaseFunctionRuntime
from layer.global_context import current_project_full_name


@ray.remote
def ray_runtime(executable_path: Path):
    LayerFunctionRuntime.execute(executable_path=Path(executable_path.name))


class RayClientFunctionRuntime(BaseFunctionRuntime):
    def __init__(self, executable_path: Path, *args, **kwargs) -> None:
        self._executable_path = executable_path
        self._address = kwargs["address"]
        self._config = ConfigManager().load()

    def initialise(self, package_info: FunctionPackageInfo) -> None:
        if not self._address:
            raise ArgumentError("Ray address is required!")
        self._packages = package_info.pip_dependencies
        self._fabric = Fabric(package_info.metadata["function"]["fabric"]["name"])
        print(f"Connecting to the Ray instance at {self._address}")
        p = [p for p in self._packages]
        p.append("layer@git+https://github.com/layerai/sdk.git@emin/ray-poc")
        self._client = ray.init(
            address=self._address,
            runtime_env={
                "working_dir": f"{self.executable_path.parent}",
                "pip": p,
                "env_vars": {
                    "LAYER_PROJECT_NAME": current_project_full_name().project_name,
                    "LAYER_CLIENT_AUTH_URL": str(self._config.url),
                    "LAYER_CLIENT_AUTH_TOKEN": self._config.client.access_token,
                },
            },
            log_to_driver=True,
            namespace="layer",
        )

    @property
    def executable_path(self) -> Path:
        return self._executable_path

    def install_packages(self, packages: Sequence[str]) -> None:
        pass

    def run_executable(self) -> Any:
        with self._client:
            ids = ray_runtime.options(
                num_cpus=self._fabric.cpu, num_gpus=self._fabric.gpu
            ).remote(self.executable_path)
            ray.wait([ids], num_returns=1)
        self._client.disconnect()
