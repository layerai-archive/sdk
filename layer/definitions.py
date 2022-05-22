import abc
import hashlib
import inspect
import os
import pickle  # nosec blacklist
import shutil
from pathlib import Path
from typing import Any, List, Union

import cloudpickle  # type: ignore

from layer.config import DEFAULT_FUNC_PATH
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import Dataset, DerivedDataset, PythonDataset
from layer.contracts.fabrics import Fabric
from layer.contracts.models import Model, Train


class Definition(abc.ABC):
    def __init__(self, func: Any, project_name: str) -> None:
        if func:
            layer_settings = func.layer
            self.func = func
            self.project_name = project_name
            self.fabric = layer_settings.get_fabric()
            self.name = layer_settings.get_entity_name()
            self.pip_packages = layer_settings.get_pip_packages()
            self.pip_requirements_file_path = layer_settings.get_pip_requirements_file()
            self.dependencies = layer_settings.get_dependencies()

    def _get_entity_dependencies(self) -> List[Union[Dataset, Model]]:
        entities = []
        for dependency in self.dependencies:
            if isinstance(dependency, Dataset) or isinstance(dependency, Model):
                entities.append(dependency)
            else:
                raise ValueError("Dependencies can only be a dataset or a model.")
        return entities

    def _get_entrypoint(self) -> str:
        return f"{self.name}.pkl"

    def _get_environment(self) -> str:
        return (
            str(os.path.basename(self.pip_requirements_file_path))
            if self.pip_requirements_file_path
            else "requirements.txt"
        )

    def _get_entity_path(self) -> Path:
        return DEFAULT_FUNC_PATH / self.project_name / self.name

    def _get_pickle_path(self) -> Path:
        return Path(f"{self._get_entity_path()}/{self._get_entrypoint()}")

    def _get_fabric(self, is_local: bool) -> str:
        if is_local:
            return Fabric.F_LOCAL.value
        else:
            return self.fabric.value if self.fabric else Fabric.default()

    def _clean_pickle_folder(self) -> None:
        # Remove directory to clean leftovers from previous runs
        entity_path = self._get_entity_path()
        if os.path.exists(entity_path):
            shutil.rmtree(entity_path)
        os.makedirs(entity_path)

    def _pack(self) -> None:
        self._clean_pickle_folder()

        # Dump pickled function to entity_name.pkl
        function_pickle_path = self._get_pickle_path()
        with open(function_pickle_path, mode="wb") as file:
            cloudpickle.dump(self.func, file, protocol=pickle.DEFAULT_PROTOCOL)

        # Add requirements to tarball
        if self.pip_requirements_file_path:
            shutil.copy(self.pip_requirements_file_path, self._get_entity_path())

        if self.pip_packages and not self.pip_requirements_file_path:
            self.pip_requirements_file_path = (
                f"{self._get_entity_path()}/{self._get_environment()}"
            )
            with open(self.pip_requirements_file_path, "w") as reqs_file:
                reqs_file.writelines(
                    list(map(lambda package: f"{package}\n", self.pip_packages))
                )

    def get_pickled_function(self) -> bytes:
        return cloudpickle.dumps(self.func, protocol=pickle.DEFAULT_PROTOCOL)

    @abc.abstractmethod
    def get_local_entity(self) -> Any:
        """Retrieve the actual entity defined in this class."""

    @abc.abstractmethod
    def get_remote_entity(self) -> Any:
        """Retrieve the actual entity defined in this class."""


class DatasetDefinition(Definition):
    def __init__(
        self,
        func: Any,
        project_name: str,
    ) -> None:
        super().__init__(func, project_name)

    def get_local_entity(self) -> DerivedDataset:
        return self.get_entity(is_local=True)

    def get_remote_entity(self) -> DerivedDataset:
        return self.get_entity(is_local=False)

    def get_entity(self, is_local: bool) -> DerivedDataset:
        super()._pack()
        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            entity_name=self.name,
            project_name=self.project_name,
        )

        return PythonDataset(
            asset_path=asset_path,
            fabric=self._get_fabric(is_local=is_local),
            entrypoint=self._get_entrypoint(),
            entrypoint_content=inspect.getsource(self.func),
            entrypoint_path=self._get_entity_path() / self._get_entrypoint(),
            environment=self._get_environment()
            if self.pip_requirements_file_path
            else "",
            environment_path=Path(self.pip_requirements_file_path)
            if self.pip_requirements_file_path
            else Path(),
            dependencies=self._get_entity_dependencies(),
        )


class ModelDefinition(Definition):
    def __init__(self, func: Any, project_name: str) -> None:
        super().__init__(
            func,
            project_name,
        )

    def get_local_entity(self) -> Model:
        return self.get_entity(is_local=True)

    def get_remote_entity(self) -> Model:
        return self.get_entity(is_local=False)

    def get_entity(self, is_local: bool) -> Model:
        super()._pack()
        train = Train(
            name=f"{self.name}-train",
            fabric=self._get_fabric(is_local=is_local),
            description="",
            entrypoint=self._get_entrypoint(),
            environment=os.path.basename(self.pip_requirements_file_path)
            if self.pip_requirements_file_path
            else "",
            hyperparameter_tuning=None,
        )

        asset_path = AssetPath(
            asset_type=AssetType.MODEL,
            entity_name=self.name,
            project_name=self.project_name,
        )

        training_files_digest = hashlib.sha256()
        training_files_digest.update(inspect.getsource(self.func).encode("utf-8"))
        model = Model(
            asset_path=asset_path,
            local_path=self._get_entity_path() / self._get_entrypoint(),
            description="",
            training=train,
            training_files_digest=training_files_digest.hexdigest(),
            parameters={},
            dependencies=self._get_entity_dependencies(),
        )
        return model
