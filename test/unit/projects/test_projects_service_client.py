import logging
import uuid
from typing import Optional
from unittest.mock import MagicMock

import pytest
from layerapi.api.entity.project_pb2 import Project
from layerapi.api.ids_pb2 import OrganizationId, ProjectId
from layerapi.api.service.flowmanager.project_api_pb2 import (
    CreateProjectRequest,
    CreateProjectResponse,
    GetProjectByPathResponse,
)

from layer.clients.project_service import ProjectServiceClient
from layer.config import ClientConfig, ProjectServiceConfig
from layer.contracts.project_full_name import ProjectFullName
from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientResourceNotFoundException,
)


def _get_project_service_client_with_mocks(
    project_api_stub: Optional[MagicMock] = None,
) -> ProjectServiceClient:
    config_mock = MagicMock(spec=ClientConfig)
    config_mock.project_service = MagicMock(spec_set=ProjectServiceConfig)
    project_service_client = ProjectServiceClient(
        config=config_mock, logger=MagicMock(spec_set=logging.getLogger())
    )
    # can"t use spec_set as it does not recognise methods as defined by protocompiler
    project_service_client._service = (  # pylint: disable=protected-access
        project_api_stub if project_api_stub is not None else MagicMock()
    )
    return project_service_client


def _get_mock_project() -> Project:
    expected_project_uuid = str(uuid.uuid4())
    proto_project_id = ProjectId(value=expected_project_uuid)
    proto_org_id = OrganizationId(value=str(uuid.uuid4()))
    return Project(
        id=proto_project_id,
        name="name",
        description="description",
        organization_id=proto_org_id,
        visibility=Project.VISIBILITY_PRIVATE,
    )


def test_given_project_exists_when_get_project_by_path_then_project_returned():
    # given
    mock_project = _get_mock_project()
    mock_project_api = MagicMock()
    mock_project_api.GetProjectByPath.return_value = GetProjectByPathResponse(
        project=mock_project
    )
    project_service_client = _get_project_service_client_with_mocks(
        project_api_stub=mock_project_api
    )

    # when
    project_full_name = ProjectFullName(
        project_name=mock_project.name, account_name="acc"
    )
    project = project_service_client.get_project(project_full_name)

    # then
    assert mock_project.id.value == str(project.id)
    assert mock_project.name == str(project.name)
    assert "acc" == str(project.account.name)
    assert mock_project.organization_id.value == str(project.account.id)


def test_given_no_project_when_get_project_by_path_then_returns_none():
    # given
    mock_project_api = MagicMock()
    mock_project_api.GetProjectByPath.side_effect = LayerClientResourceNotFoundException
    project_service_client = _get_project_service_client_with_mocks(
        project_api_stub=mock_project_api
    )

    # when
    project_full_name = ProjectFullName(project_name="test", account_name="acc")
    project = project_service_client.get_project(project_full_name)

    # then
    assert project is None


def test_given_unknown_error_when_get_project_by_path_raises_unhandled_grpc_error():  # noqa
    # given
    mock_project_api = MagicMock()
    mock_project_api.GetProjectByPath.side_effect = Exception
    project_service_client = _get_project_service_client_with_mocks(
        project_api_stub=mock_project_api
    )

    # when + then
    project_full_name = ProjectFullName(project_name="test", account_name="acc")
    with pytest.raises(LayerClientException):
        project_service_client.get_project(project_full_name)


def test_given_project_not_exists_when_update_project_raise_resource_not_found_error():  # noqa
    # given
    mock_project_api = MagicMock()
    mock_project_api.UpdateProject.side_effect = LayerClientResourceNotFoundException
    project_service_client = _get_project_service_client_with_mocks(
        project_api_stub=mock_project_api
    )

    # when + then
    with pytest.raises(LayerClientResourceNotFoundException):
        project_service_client.update_project_readme("name", "readme")


def test_given_project_not_exists_when_create_project_creates_project_with_private_visibility():  # noqa
    # given
    mock_project = _get_mock_project()
    mock_project_api = MagicMock()
    mock_project_api.CreateProject.return_value = CreateProjectResponse(
        project=mock_project
    )
    project_service_client = _get_project_service_client_with_mocks(
        project_api_stub=mock_project_api
    )

    # when
    project_full_name = ProjectFullName(project_name="test", account_name="acc")
    project_service_client.create_project(project_full_name)

    # then
    mock_project_api.CreateProject.assert_called_with(
        CreateProjectRequest(project_name="test", visibility=Project.VISIBILITY_PRIVATE)
    )
