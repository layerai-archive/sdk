import hashlib
import inspect
import os
import pickle  # nosec import_pickle
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import cloudpickle  # type: ignore

from layer.config import DEFAULT_FUNC_PATH
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName


@dataclass(frozen=True)
class ResourcePath:
    # Local file system path of the resource (file or dir), relative to the project dir.
    # Examples: data/test.csv, users.parquet
    path: str

    def local_relative_paths(self) -> Iterator[str]:
        """
        Map path to the absolute file system paths, checking if paths exist.
        Includes single files and files in each resource directory.

        :return: iterator of absolute file paths.
        """
        file_path = os.path.relpath(self.path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"resource file or directory: {self.path} in {os.getcwd()}"
            )
        if os.path.isfile(file_path):
            yield file_path
        if os.path.isdir(file_path):
            for root, _, files in os.walk(file_path):
                for f in files:
                    dir_file_path = os.path.join(root, f)
                    yield os.path.relpath(dir_file_path)


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

        self._pack()

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

    def with_version_id(self, version_id: uuid.UUID) -> "FunctionDefinition":
        self.version_id = version_id
        return self

    def with_repository_id(self, repository_id: uuid.UUID) -> "FunctionDefinition":
        self.repository_id = repository_id
        return self

    def drop_dependencies(self) -> "FunctionDefinition":
        self.asset_dependencies = []
        return self

    # DEPRECATED below
    def get_pickled_function(self) -> bytes:
        return cloudpickle.dumps(self.func, protocol=pickle.DEFAULT_PROTOCOL)

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

    def _clean_pickle_folder(self) -> None:
        # Remove directory to clean leftovers from previous runs
        function_home_dir = self.function_home_dir
        if function_home_dir.exists():
            shutil.rmtree(function_home_dir)
        os.makedirs(function_home_dir)

    def _pack(self) -> None:
        self._clean_pickle_folder()

        # Dump pickled function to asset_name.pkl
        with open(self.pickle_path, mode="wb") as file:
            cloudpickle.dump(self.func, file, protocol=pickle.DEFAULT_PROTOCOL)

        with open(self.environment_path, "w") as reqs_file:
            reqs_file.write("\n".join(self.pip_dependencies))
