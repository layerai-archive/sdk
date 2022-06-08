from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectFullName:
    """
    Represents the project name and its owning account name
    """

    project_name: str
    account_name: str

    @property
    def path(self) -> str:
        return f"{self.account_name}/{self.project_name}"
