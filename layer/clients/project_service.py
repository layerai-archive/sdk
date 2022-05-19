import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from typing import Iterator, Optional
from uuid import UUID

from layerapi.api.entity.project_pb2 import Project
from layerapi.api.ids_pb2 import ProjectId
from layerapi.api.service.flowmanager.project_api_pb2 import (
    CreateProjectRequest,
    GetProjectByNameRequest,
    RemoveProjectByIdRequest,
    UpdateProjectRequest,
)
from layerapi.api.service.flowmanager.project_api_pb2_grpc import ProjectAPIStub

from layer.config import ClientConfig
from layer.exceptions.exceptions import (
    LayerClientResourceAlreadyExistsException,
    LayerClientResourceNotFoundException,
)
from layer.utils.grpc import create_grpc_channel, generate_client_error_from_grpc_error


@dataclass(frozen=True)
class ProjectIdWithAccountId:
    project_id: Optional[UUID] = None
    account_id: Optional[UUID] = None


class ProjectServiceClient:
    _service: ProjectAPIStub

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

    @contextmanager
    def init(self) -> Iterator["ProjectServiceClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = ProjectAPIStub(channel=channel)
            yield self

    def get_project_id_and_org_id(self, name: str) -> ProjectIdWithAccountId:
        project_id_with_org_id = ProjectIdWithAccountId()
        try:
            resp = self._service.GetProjectByName(
                GetProjectByNameRequest(project_name=name)
            )
            if resp.project is not None:
                project_id_with_org_id = ProjectIdWithAccountId(
                    project_id=UUID(resp.project.id.value),
                    account_id=UUID(resp.project.organization_id.value),
                )
        except LayerClientResourceNotFoundException:
            pass
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
        return project_id_with_org_id

    def remove_project(self, project_id: uuid.UUID) -> None:
        self._service.RemoveProjectById(
            RemoveProjectByIdRequest(project_id=ProjectId(value=str(project_id)))
        )

    def create_project(self, name: str) -> ProjectIdWithAccountId:
        try:
            resp = self._service.CreateProject(
                CreateProjectRequest(
                    project_name=name, visibility=Project.VISIBILITY_PRIVATE
                )
            )
            return ProjectIdWithAccountId(
                project_id=UUID(resp.project.id.value),
                account_id=UUID(resp.project.organization_id.value),
            )
        except LayerClientResourceAlreadyExistsException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def update_project_readme(self, project_name: str, readme: str) -> None:
        try:
            self._service.UpdateProject(
                UpdateProjectRequest(project_name=project_name, readme=readme)
            )
        except LayerClientResourceNotFoundException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def set_project_visibility(self, project_name: str, *, is_public: bool) -> None:
        visibility = (
            Project.VISIBILITY_PUBLIC if is_public else Project.VISIBILITY_PRIVATE
        )
        try:
            self._service.UpdateProject(
                UpdateProjectRequest(project_name=project_name, visibility=visibility)
            )
        except LayerClientResourceNotFoundException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
