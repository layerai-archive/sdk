import logging
import os
import uuid
import warnings
from collections import namedtuple
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Set

from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.ids_pb2 import HyperparameterTuningId, ModelVersionId

from layer.contracts.accounts import Account
from layer.contracts.asset import AssetType
from layer.contracts.datasets import DerivedDataset, RawDataset
from layer.contracts.models import Model
from layer.exceptions.exceptions import ProjectException


logger = logging.getLogger()
LeveledNode = namedtuple("LeveledNode", ["node", "level"])


@dataclass(frozen=True)
class Asset:
    type: AssetType
    name: str


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


@dataclass(frozen=True)
class Function:
    name: str
    asset: Asset
    resource_paths: Set[ResourcePath] = field(default_factory=set)


@dataclass(frozen=True)
class Project:
    """
    Provides access to projects stored in Layer. Projects are containers to organize your machine learning project assets.

    You can retrieve an instance of this object with :code:`layer.init()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Initializes a project with name "my-project"
        layer.init("my-project")

    """

    name: str
    raw_datasets: Sequence[RawDataset] = field(default_factory=list)
    derived_datasets: Sequence[DerivedDataset] = field(default_factory=list)
    models: Sequence[Model] = field(default_factory=list)
    path: Path = field(compare=False, default=Path())
    project_files_hash: str = ""
    readme: str = ""
    account: Optional[Account] = None
    _id: Optional[uuid.UUID] = None
    functions: Sequence[Function] = field(default_factory=list)

    @property
    def id(self) -> uuid.UUID:
        if self._id is None:
            raise ProjectException("project has no id defined")
        return self._id

    def with_name(self, name: str) -> "Project":
        """
        :return: A new object that has a new name but all other fields are the same.
        """
        return replace(self, name=name)

    def with_id(self, project_id: uuid.UUID) -> "Project":
        """
        :return: A new object that has a new id but all other fields are the same.
        """
        return replace(self, _id=project_id)

    def with_account(self, account: Account) -> "Project":
        return replace(self, account=account)

    def with_raw_datasets(self, raw_datasets: Iterable[RawDataset]) -> "Project":
        return replace(self, raw_datasets=list(raw_datasets))

    def with_derived_datasets(
        self, derived_datasets: Iterable[DerivedDataset]
    ) -> "Project":
        return replace(self, derived_datasets=list(derived_datasets))

    def with_models(self, models: Iterable[Model]) -> "Project":
        return replace(self, models=list(models))

    def with_path(self, path: Path) -> "Project":
        return replace(self, path=path)

    def with_files_hash(self, new_hash: str) -> "Project":
        return replace(self, project_files_hash=new_hash)

    def with_readme(self, readme: Optional[str]) -> "Project":
        return replace(self, readme=readme)

    def with_functions(self, functions: Sequence[Function]) -> "Project":
        return replace(self, functions=functions)


@dataclass(frozen=True)
class ApplyResult:
    execution_plan: ExecutionPlan
    models_metadata: Dict[str, ModelVersionId] = field(default_factory=dict)
    hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId] = field(
        default_factory=dict
    )

    def with_models_metadata(
        self,
        models_metadata: Dict[str, ModelVersionId],
        hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId],
        plan: ExecutionPlan,
    ) -> "ApplyResult":
        return replace(
            self,
            models_metadata=dict(models_metadata),
            hyperparameter_tuning_metadata=dict(hyperparameter_tuning_metadata),
            execution_plan=plan,
        )


class ProjectLoader:
    @staticmethod
    def load_project_readme(path: Path) -> Optional[str]:
        for subp in path.glob("*"):
            if subp.name.lower() == "readme.md":
                with open(subp, "r") as f:
                    readme = f.read()
                    # restrict length of text we send to backend
                    if len(readme) > 25_000:
                        warnings.warn(
                            "Your README.md will be truncated to 25000 characters",
                        )
                        readme = readme[:25_000]
                    return readme
        return None
