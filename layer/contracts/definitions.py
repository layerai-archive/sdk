import hashlib
import inspect
import os
import pickle  # nosec import_pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, List, Optional

from layer.config import DEFAULT_FUNC_PATH, is_executables_feature_active
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import ResourcePath
from layer.executables.packager import package_function

from .. import cloudpickle


class FunctionDefinition:
    def __init__(
        self,
        func: Callable[..., Any],
        project_name: str,
        account_name: str,
        asset_type: AssetType,
        asset_name: str,
        fabric: Fabric,
        asset_dependencies: List[AssetPath],
        pip_dependencies: List[str],
        resource_paths: List[ResourcePath],
        assertions: List[Assertion],
        version_id: Optional[uuid.UUID] = None,
        repository_id: Optional[uuid.UUID] = None,
        description: str = "",
        uri: str = "",
    ) -> None:
        self.func = func
        self.func_name: str = func.__name__

        self.project_name = project_name
        self.account_name = account_name
        self.asset_type = asset_type
        self.asset_name = asset_name
        self.fabric = fabric
        self.asset_dependencies = asset_dependencies
        self.pip_dependencies = pip_dependencies
        self.resource_paths = resource_paths
        self.assertions = assertions

        self.version_id = version_id
        self.repository_id = repository_id
        self.description = description
        self.uri = uri

        self.func_source = inspect.getsource(self.func)

        self.source_code_digest = hashlib.sha256()
        self.source_code_digest.update(self.func_source.encode("utf-8"))

        self._executable_path: Optional[Path] = None

    def __repr__(self) -> str:
        return f"FunctionDefinition({self.asset_type}, {self.asset_name})"

    @property
    def project_full_name(self) -> ProjectFullName:
        return ProjectFullName(
            account_name=self.account_name,
            project_name=self.project_name,
        )

    @property
    def asset_path(self) -> AssetPath:
        return AssetPath(
            asset_type=self.asset_type,
            asset_name=self.asset_name,
            project_name=self.project_name,
            org_name=self.account_name,
        )

    @property
    def function_home_dir(self) -> Path:
        return DEFAULT_FUNC_PATH / self.asset_path.path()

    def set_version_id(self, version_id: uuid.UUID) -> None:
        self.version_id = version_id

    def set_repository_id(self, repository_id: uuid.UUID) -> None:
        self.repository_id = repository_id

    # DEPRECATED below, will remove once we build the simplified backend
    def get_pickled_function(self) -> bytes:
        return cloudpickle.dumps(self.func, protocol=pickle.DEFAULT_PROTOCOL)  # type: ignore

    @property
    def entrypoint(self) -> str:
        return f"{self.asset_name}.pkl"

    @property
    def pickle_path(self) -> Path:
        return self.function_home_dir / self.entrypoint

    @property
    def environment(self) -> str:
        return "requirements.txt"

    @property
    def environment_path(self) -> Path:
        return self.function_home_dir / self.environment

    def get_fabric(self, is_local: bool) -> str:
        if is_local:
            return Fabric.F_LOCAL.value
        else:
            return self.fabric.value

    def _clean_function_home_dir(self) -> None:
        # Remove directory to clean leftovers from previous runs
        function_home_dir = self.function_home_dir
        if function_home_dir.exists():
            shutil.rmtree(function_home_dir)
        os.makedirs(function_home_dir)

    def package(self) -> Path:
        self._clean_function_home_dir()
        if is_executables_feature_active():
            self._executable_path = self._package_executable()
            return self._executable_path
        else:
            # Dump pickled function to asset_name.pkl
            with open(self.pickle_path, mode="wb") as file:
                cloudpickle.dump(self.func, file, protocol=pickle.DEFAULT_PROTOCOL)  # type: ignore

            with open(self.environment_path, "w") as reqs_file:
                reqs_file.write("\n".join(self.pip_dependencies))

            return self.pickle_path

    def _package_executable(self) -> Path:
        resource_paths = [Path(resource.path) for resource in self.resource_paths]
        return package_function(
            self.func,
            resources=resource_paths,
            pip_dependencies=self.pip_dependencies,
            output_dir=self.function_home_dir,
        )

    @property
    def executable_path(self) -> Path:
        if self._executable_path is None:
            self._executable_path = self._package_executable()
        return self._executable_path
