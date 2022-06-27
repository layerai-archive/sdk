import uuid
from logging import Logger
from typing import Optional
from uuid import UUID

from layer import Project
from layer.config import ClientConfig
from layer.contracts.accounts import Account
from layer.contracts.project_full_name import ProjectFullName


class ProjectServiceClient:
    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.project_service
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    def get_project_by_id(self, project_id: UUID) -> Optional[Project]:
        pass

    def get_project(self, full_name: ProjectFullName) -> Optional[Project]:
        return Project(
            account=Account(id=uuid.uuid4(), name="my_user"),
            name=full_name.project_name,
            id=uuid.uuid4(),
        )

    def remove_project(self, project_id: uuid.UUID) -> None:
        pass

    def create_project(self, full_name: ProjectFullName) -> Project:
        pass

    def update_project_readme(
        self, project_full_name: ProjectFullName, readme: str
    ) -> None:
        pass

    def set_project_visibility(
        self, project_full_name: ProjectFullName, *, is_public: bool
    ) -> None:
        pass
