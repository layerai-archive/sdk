import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from typing import Iterator, Optional
from uuid import UUID

from layerapi.api.entity.project_pb2 import Project
from layerapi.api.ids_pb2 import AccountId, ProjectId
from layerapi.api.service.account.account_api_pb2 import (
    GetAccountViewByIdRequest,
    GetAccountViewByIdResponse,
)
from layerapi.api.service.account.account_api_pb2_grpc import AccountAPIStub
from layerapi.api.service.flowmanager.project_api_pb2 import (
    CreateProjectRequest,
    GetProjectByIdRequest,
    GetProjectByIdResponse,
    GetProjectByNameRequest,
    RemoveProjectByIdRequest,
    UpdateProjectRequest,
)
from layerapi.api.service.flowmanager.project_api_pb2_grpc import ProjectAPIStub

from layer.config import ClientConfig
from layer.contracts.accounts import Account
from layer.contracts.projects import Project as ProjectView
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
    _account_stub: AccountAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.project_service
        self._account_config = config.account_service
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

        with create_grpc_channel(
            self._account_config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._account_stub = AccountAPIStub(channel=channel)

        yield self

    def get_project(self, project_id: uuid.UUID) -> Optional[ProjectView]:
        try:
            resp: GetProjectByIdResponse = self._service.GetProjectById(
                GetProjectByIdRequest(project_id=ProjectId(value=str(project_id)))
            )
            if resp.project is not None:
                account_id = UUID(resp.project.organization_id.value)
                account_resp: GetAccountViewByIdResponse = (
                    self._account_stub.GetAccountViewById(
                        GetAccountViewByIdRequest(id=AccountId(value=str(account_id)))
                    )
                )
                account = Account(
                    id=account_id,
                    name=account_resp.account_view.name,
                )
                return ProjectView(
                    _id=project_id, name=resp.project.name, account=account
                )
        except LayerClientResourceNotFoundException:
            pass
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
        return None

    def get_project_id(self, account: Account, name: str) -> Optional[UUID]:
        try:
            resp = self._service.GetProjectByName(
                GetProjectByNameRequest(
                    account_id=AccountId(value=str(account.id)), project_name=name
                )
            )
            if resp.project is not None:
                return UUID(resp.project.id.value)
        except LayerClientResourceNotFoundException:
            return None
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def remove_project(self, project_id: uuid.UUID) -> None:
        self._service.RemoveProjectById(
            RemoveProjectByIdRequest(project_id=ProjectId(value=str(project_id)))
        )

    def create_project(self, parent_account: Account, project_name: str) -> UUID:
        try:
            resp = self._service.CreateProject(
                CreateProjectRequest(
                    # account_id=AccountId(value=str(parent_account.id)),
                    project_name=project_name,
                    visibility=Project.VISIBILITY_PRIVATE,
                )
            )
            return UUID(resp.project.id.value)
        except LayerClientResourceAlreadyExistsException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def update_project_readme(self, project_id: uuid.UUID, readme: str) -> None:
        try:
            self._service.UpdateProject(
                # TODO Change project api
                UpdateProjectRequest(project_id=project_id, readme=readme)
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
