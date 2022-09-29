import os
import pathlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ResourcePath:
    # Local file system path of the resource (file or dir), relative to the project dir.
    # Examples: data/test.csv, users.parquet
    path: pathlib.Path

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
