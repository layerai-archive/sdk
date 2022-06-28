import os
from dataclasses import dataclass
from typing import Callable, Iterator, List

from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.ids_pb2 import RunId

from layer.contracts.project_full_name import ProjectFullName


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

    id: RunId
    project_full_name: ProjectFullName

    @property
    def project_name(self) -> str:
        return self.project_full_name.project_name

    @property
    def account_name(self) -> str:
        return self.project_full_name.account_name
