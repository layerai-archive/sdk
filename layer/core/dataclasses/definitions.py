import hashlib
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from layer.config import DEFAULT_FUNC_PATH
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.conda import CondaEnv
from layer.contracts.fabrics import Fabric
from layer.contracts.runs import ResourcePath
from layer.executables.packager import package_function
from layer.runs import context


class FunctionDefinition:
    def __init__(
        self,
        func: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        asset_type: AssetType,
        asset_name: str,
        fabric: Fabric,
        asset_dependencies: List[AssetPath],
        pip_dependencies: List[str],
        conda_env: Optional[CondaEnv],
        resource_paths: List[ResourcePath],
        assertions: List[Assertion],
        package_download_url: Optional[str] = None,
        description: str = "",
        uri: str = "",
    ) -> None:
        self.func = func
        self.func_name: str = func.__name__
        self.args = args
        self.kwargs = kwargs

        self.asset_type = asset_type
        self.asset_name = asset_name
        self.description = description
        self.fabric = fabric
        self.asset_dependencies = asset_dependencies
        self.pip_dependencies = pip_dependencies
        self.conda_env = conda_env
        self.resource_paths = resource_paths
        self.assertions = assertions

        self.package_download_url = package_download_url
        self.uri = uri

        self.func_source = self._get_source()

        source_code_digest = hashlib.sha256()
        source_code_digest.update(self.func_source.encode("utf-8"))
        self.source_code_digest = source_code_digest.hexdigest()

        self._executable_path: Optional[Path] = None

    def __repr__(self) -> str:
        return f"FunctionDefinition({self.asset_type}, {self.asset_name})"

    def _get_source(self) -> str:
        try:
            return inspect.getsource(self.func)
        except Exception:
            return "source code not available"

    @property
    def asset_path(self) -> AssetPath:
        project_full_name = context.get_project_full_name()
        return AssetPath(
            asset_type=self.asset_type,
            asset_name=self.asset_name,
            project_name=project_full_name.project_name,
            account_name=project_full_name.account_name,
        )

    @property
    def function_home_dir(self) -> Path:
        return DEFAULT_FUNC_PATH / self.asset_path.path()

    def set_package_download_url(self, package_download_url: str) -> None:
        self.package_download_url = package_download_url

    def package(self) -> Path:
        self._clean_function_home_dir()
        self._executable_path = self._package_executable()
        return self._executable_path

    def _package_executable(self) -> Path:
        resource_paths = [Path(resource.path) for resource in self.resource_paths]
        return package_function(
            self.runner_function(),
            resources=resource_paths,
            pip_dependencies=self.pip_dependencies,
            conda_env=self.conda_env,
            output_dir=self.function_home_dir,
        )

    def runner_function(self) -> Any:
        if self.asset_type == AssetType.DATASET:
            from layer.executables.entrypoint.dataset import DatasetRunner

            return DatasetRunner(self, context.get_run_context())
        elif self.asset_type == AssetType.MODEL:
            from layer.executables.entrypoint.model import ModelRunner

            return ModelRunner(self, context.get_run_context())
        raise Exception(f"Invalid asset type {self.asset_type}")

    def _clean_function_home_dir(self) -> None:
        # Remove directory to clean leftovers from previous runs
        function_home_dir = self.function_home_dir
        if function_home_dir.exists():
            shutil.rmtree(function_home_dir)
        os.makedirs(function_home_dir)

    @property
    def executable_path(self) -> Path:
        if self._executable_path is None:
            self._executable_path = self._package_executable()
        return self._executable_path
