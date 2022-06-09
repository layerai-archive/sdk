from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectFullName:
    """
    Represents the project name and its owning account name
    """

    account_name: str
    project_name: str

    @property
    def path(self) -> str:
        return f"{self.account_name}/{self.project_name}"
