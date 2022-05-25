import logging
import uuid
import warnings
from collections import namedtuple
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from layerapi.api.entity.operations_pb2 import ExecutionPlan

from layer.exceptions.exceptions import ProjectException

from .accounts import Account
from .asset import AssetType


logger = logging.getLogger()
LeveledNode = namedtuple("LeveledNode", ["node", "level"])


@dataclass(frozen=True)
class Asset:
    type: AssetType
    name: str


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
    account: Optional[Account] = None
    _id: Optional[uuid.UUID] = None

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

    def with_path(self, path: Path) -> "Project":
        return replace(self, path=path)

    def with_files_hash(self, new_hash: str) -> "Project":
        return replace(self, project_files_hash=new_hash)

    def with_readme(self, readme: Optional[str]) -> "Project":
        return replace(self, readme=readme)


@dataclass(frozen=True)
class ApplyResult:
    execution_plan: ExecutionPlan


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
