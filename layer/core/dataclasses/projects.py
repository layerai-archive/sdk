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
from .project_full_name import ProjectFullName


logger = logging.getLogger()
LeveledNode = namedtuple("LeveledNode", ["node", "level"])


@dataclass(frozen=True)
class Project:
    """
    Provides access to projects stored in Layer. Projects are containers to organize your machine learning project entities.

    You can retrieve an instance of this object with :code:`layer.init()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Initializes a project with name "my-project"
        layer.init("my-project")

    """

    name: str
    id: uuid.UUID
    account: Account

    @property
    def full_name(self) -> ProjectFullName:
        if self.account is None:
            raise ProjectException("project has no account defined")
        return ProjectFullName(
            project_name=self.name,
            account_name=self.account.name,
        )

    def with_name(self, name: str) -> "Project":
        """
        :return: A new object that has a new name but all other fields are the same.
        """
        return replace(self, name=name)

    def with_id(self, project_id: uuid.UUID) -> "Project":
        """
        :return: A new object that has a new id but all other fields are the same.
        """
        return replace(self, id=project_id)

    def with_account(self, account: Account) -> "Project":
        return replace(self, account=account)

    def __str__(self) -> str:
        if self.account:
            from layer.config import DEFAULT_PATH, ConfigManager

            config = ConfigManager(DEFAULT_PATH).load()
            return f"Your Layer project is here: {config.url}/{self.account.name}/{self.name}"
        else:
            return f"Project({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


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
