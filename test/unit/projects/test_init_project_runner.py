import logging
import uuid
from typing import Optional
from unittest.mock import MagicMock, Mock

from layer import current_project_name
from layer.common import LayerClient
from layer.config import ConfigManager
from layer.contracts.fabrics import Fabric
from layer.global_context import (
    default_fabric,
    get_pip_packages,
    get_pip_requirements_file,
)
from layer.projects import ProjectServiceClient
from layer.projects.client import ProjectIdWithAccountId
from layer.projects.init_project_runner import InitProjectRunner


def _get_random_project_id_with_org_id():
    expected_project_id = uuid.uuid4()
    expected_account_id = uuid.uuid4()
    return ProjectIdWithAccountId(
        project_id=expected_project_id, account_id=expected_account_id
    )


def _get_mock_layer_client(
    project_client_mock: Optional[MagicMock] = None,
) -> MagicMock:
    project_client_mock = project_client_mock or MagicMock(
        spec_set=ProjectServiceClient
    )
    layer_client_mock = MagicMock(
        spec_set=LayerClient,
        project_service_client=project_client_mock,
    )
    layer_client_mock.init.return_value.__enter__.return_value = layer_client_mock
    return layer_client_mock


def test_given_project_exists_when_set_up_project_gets_and_sets_global_project():
    # given
    expected_project_id_with_org_id = _get_random_project_id_with_org_id()
    expected_project_name = "project_name"
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project_id_and_org_id=Mock(
            return_value=expected_project_id_with_org_id,
            spec_set=ProjectServiceClient.get_project_id_and_org_id,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    my_fabric = Fabric.F_MEDIUM
    pip_requirements_file_name = "req.txt"
    # when
    project_runner = InitProjectRunner(
        project_name=expected_project_name,
        logger=MagicMock(spec_set=logging.getLogger()),
        config_manager=MagicMock(spec_set=ConfigManager),
    )
    project = project_runner.setup_project(
        layer_client_mock,
        fabric=my_fabric,
        pip_requirements_file=pip_requirements_file_name,
    )

    # then
    assert project._id == expected_project_id_with_org_id.project_id
    assert current_project_name() == expected_project_name
    assert default_fabric() == my_fabric
    assert get_pip_requirements_file() == pip_requirements_file_name
    project_client_mock.get_project_id_and_org_id.assert_called_once()
    project_client_mock.create_project.assert_not_called()


def test_given_project_not_exists_when_set_up_project_creates_and_sets_global_project():
    # given
    missing_project_id_with_org_id = ProjectIdWithAccountId()
    expected_project_id_with_org_id = _get_random_project_id_with_org_id()
    expected_project_name = "project_name"
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project_id_and_org_id=Mock(
            return_value=missing_project_id_with_org_id,
            spec_set=ProjectServiceClient.get_project_id_and_org_id,
        ),
        create_project=Mock(
            return_value=expected_project_id_with_org_id,
            spec_set=ProjectServiceClient.create_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    my_fabric = Fabric.F_MEDIUM
    pip_packages = ["sklearn==0.0"]

    # when
    project_runner = InitProjectRunner(
        project_name=expected_project_name,
        logger=MagicMock(spec_set=logging.getLogger()),
        config_manager=MagicMock(spec_set=ConfigManager),
    )
    project = project_runner.setup_project(
        layer_client_mock,
        fabric=my_fabric,
        pip_packages=pip_packages,
    )

    # then
    assert project._id == expected_project_id_with_org_id.project_id
    assert current_project_name() == expected_project_name
    assert default_fabric() == my_fabric
    assert get_pip_packages() == pip_packages
    project_client_mock.get_project_id_and_org_id.assert_called_once()
    project_client_mock.create_project.assert_called_once()


def test_given_readme_exists_when_set_up_project_gets_and_sets_project_readme(tmp_path):
    # given
    readme_path = tmp_path / "README.md"

    expected_readme = "Sample README"
    with open(readme_path, "w") as f:
        f.write(expected_readme)

    expected_project_id_with_org_id = _get_random_project_id_with_org_id()
    expected_project_name = "project_name"
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project_id_and_org_id=Mock(
            return_value=expected_project_id_with_org_id,
            spec_set=ProjectServiceClient.get_project_id_and_org_id,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)

    # when
    project_runner = InitProjectRunner(
        project_root_path=tmp_path,
        project_name=expected_project_name,
        logger=MagicMock(spec_set=logging.getLogger()),
        config_manager=MagicMock(spec_set=ConfigManager),
    )
    project_runner.setup_project(
        layer_client_mock,
    )

    # then
    layer_client_mock.project_service_client.update_project_readme.assert_called_with(
        project_name=expected_project_name, readme=expected_readme
    )


def test_given_readme_not_exists_when_set_up_project_gets_and_setup_project(tmp_path):
    # given
    expected_project_id_with_org_id = _get_random_project_id_with_org_id()
    expected_project_name = "project_name"
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project_id_and_org_id=Mock(
            return_value=expected_project_id_with_org_id,
            spec_set=ProjectServiceClient.get_project_id_and_org_id,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)

    # when
    project_runner = InitProjectRunner(
        project_root_path=tmp_path,
        project_name=expected_project_name,
        logger=MagicMock(spec_set=logging.getLogger()),
        config_manager=MagicMock(spec_set=ConfigManager),
    )
    project = project_runner.setup_project(
        layer_client_mock,
    )

    # then
    assert project.name == expected_project_name


def test_given_long_readme_exists_when_set_up_project_gets_and_sets_project_readme(
    tmp_path,
):
    # given
    readme_path = tmp_path / "ReAdmE.Md"

    actual_readme = "".join("." for i in range(30_000))
    expected_readme = actual_readme[:25_000]
    with open(readme_path, "w") as f:
        f.write(actual_readme)

    expected_project_id_with_org_id = _get_random_project_id_with_org_id()
    expected_project_name = "project_name"
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project_id_and_org_id=Mock(
            return_value=expected_project_id_with_org_id,
            spec_set=ProjectServiceClient.get_project_id_and_org_id,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    # when
    project_runner = InitProjectRunner(
        project_root_path=tmp_path,
        project_name=expected_project_name,
        logger=MagicMock(spec_set=logging.getLogger()),
        config_manager=MagicMock(spec_set=ConfigManager),
    )
    project_runner.setup_project(
        layer_client_mock,
    )

    # then
    layer_client_mock.project_service_client.update_project_readme.assert_called_with(
        project_name=expected_project_name, readme=expected_readme
    )
