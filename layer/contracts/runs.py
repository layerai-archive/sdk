import abc
import hashlib
import inspect
import os
import pickle  # nosec import_pickle
import shutil
import sys
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import cloudpickle  # type: ignore
from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.ids_pb2 import RunId

from layer.config import DEFAULT_FUNC_PATH
from layer.exceptions.exceptions import LayerClientException

from .accounts import Account
from .asset import AssetPath, AssetType
from .fabrics import Fabric


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro


GetRunsFunction = Callable[[], List[PBRun]]


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


FunctionDefinitionType = TypeVar(  # pylint: disable=invalid-name
    "FunctionDefinitionType", bound="FunctionDefinition"
)


class FunctionDefinition(abc.ABC):
    def __init__(
        self,
        func: Any,
        project_name: str,
        account_name: Optional[str] = None,
        version_id: Optional[uuid.UUID] = None,
        repository_id: Optional[uuid.UUID] = None,
        description: str = "",
        uri: str = "",
        language_version: Tuple[int, int, int] = _language_version(),
    ) -> None:
        self.func = func
        self.func_name: str = func.__name__
        self.project_name = project_name
        self.account_name = account_name
        self.version_id = version_id
        self.repository_id = repository_id
        self.description = description
        self.uri = uri
        self.language_version = language_version

        layer_settings = func.layer
        name = layer_settings.get_entity_name()
        if name is None:
            raise LayerClientException("Name cannot be empty")
        self.name = name
        self.resource_paths = {
            ResourcePath(path=path) for path in layer_settings.get_paths() or []
        }
        fabric = layer_settings.get_fabric()
        if fabric is None:
            fabric = Fabric.default()
        self._fabric = fabric
        self._dependencies = layer_settings.get_dependencies()
        self.pip_packages = layer_settings.get_pip_packages()
        self.pip_requirements_file = layer_settings.get_pip_requirements_file()

        self.func_source = inspect.getsource(self.func)

        self.source_code_digest = hashlib.sha256()
        self.source_code_digest.update(self.func_source.encode("utf-8"))

        self._pack()

    def __repr__(self) -> str:
        return f"FunctionDefinition({self.asset_type}, {self.name})"

    @abc.abstractproperty
    def asset_type(self) -> AssetType:
        ...

    @property
    def asset_path(self) -> AssetPath:
        return AssetPath(
            asset_type=self.asset_type,
            entity_name=self.name,
            project_name=self.project_name,
        )

    @property
    def dependencies(self) -> List[AssetPath]:
        return [d.with_project_name(self.project_name) for d in self._dependencies]

    @property
    def entrypoint(self) -> str:
        return f"{self.name}.pkl"

    @property
    def entity_path(self) -> Path:
        return DEFAULT_FUNC_PATH / self.project_name / self.name

    @property
    def pickle_path(self) -> Path:
        return self.entity_path / self.entrypoint

    @property
    def environment(self) -> str:
        return (
            str(os.path.basename(self.pip_requirements_file))
            if self.pip_requirements_file
            else "requirements.txt"
        )

    @property
    def environment_path(self) -> Path:
        return self.entity_path / self.environment

    def get_fabric(self, is_local: bool) -> str:
        if is_local:
            return Fabric.F_LOCAL.value
        else:
            return self._fabric.value

    def _clean_pickle_folder(self) -> None:
        # Remove directory to clean leftovers from previous runs
        entity_path = self.entity_path
        if entity_path.exists():
            shutil.rmtree(entity_path)
        os.makedirs(entity_path)

    def _pack(self) -> None:
        self._clean_pickle_folder()

        # Dump pickled function to entity_name.pkl
        with open(self.pickle_path, mode="wb") as file:
            cloudpickle.dump(self.func, file, protocol=pickle.DEFAULT_PROTOCOL)

        # Add requirements to tarball
        if self.pip_requirements_file:
            shutil.copy(self.pip_requirements_file, self.environment_path)
        elif self.pip_packages:
            with open(self.environment_path, "w") as reqs_file:
                reqs_file.writelines(
                    list(map(lambda package: f"{package}\n", self.pip_packages))
                )

    def get_pickled_function(self) -> bytes:
        return cloudpickle.dumps(self.func, protocol=pickle.DEFAULT_PROTOCOL)

    def with_version_id(
        self: FunctionDefinitionType, version_id: uuid.UUID
    ) -> FunctionDefinitionType:
        self.version_id = version_id
        return self

    def with_repository_id(
        self: FunctionDefinitionType, repository_id: uuid.UUID
    ) -> FunctionDefinitionType:
        self.repository_id = repository_id
        return self

    def set_account_name(
        self: FunctionDefinitionType, account_name: str
    ) -> FunctionDefinitionType:
        self.account_name = account_name
        return self

    def drop_dependencies(self: FunctionDefinitionType) -> FunctionDefinitionType:
        self._dependencies = set()
        return self


class DatasetFunctionDefinition(FunctionDefinition):
    """
    Dataset function definition
    """

    @property
    def asset_type(self) -> AssetType:
        return AssetType.DATASET


class ModelFunctionDefinition(FunctionDefinition):
    """
    Model function definition
    """

    @property
    def asset_type(self) -> AssetType:
        return AssetType.MODEL


@dataclass(frozen=True)
class Run:
    """
    Provides access to project runs stored in Layer.

    You can retrieve an instance of this object with :code:`layer.run()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Runs the current project with the given functions
        layer.run([build_dataset, train_model])

    """

    project_id: uuid.UUID = field(repr=False)
    project_name: str
    definitions: Sequence[FunctionDefinition] = field(default_factory=list, repr=False)
    files_hash: str = ""
    account: Optional[Account] = None
    readme: str = field(repr=False, default="")
    run_id: Optional[RunId] = field(repr=False, default=None)

    def with_run_id(self, run_id: RunId) -> "Run":
        return replace(self, run_id=run_id)

    def with_definitions(self, definitions: Sequence[FunctionDefinition]) -> "Run":
        return replace(self, definitions=definitions)

    def with_account(self, account: Account) -> "Run":
        return replace(self, account=account)
