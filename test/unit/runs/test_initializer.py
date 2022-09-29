import uuid
from typing import Optional
from unittest.mock import MagicMock, Mock

from yarl import URL

from layer.clients.layer import LayerClient
from layer.clients.project_service import ProjectServiceClient
from layer.contracts.accounts import Account
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.runs import context
from layer.runs.initializer import RunInitializer


def _get_random_project(
    full_name: ProjectFullName = ProjectFullName("p-test", "atest")
) -> Project:
    project_id = uuid.uuid4()
    account_id = uuid.uuid4()
    return Project(
        id=project_id,
        name=full_name.project_name,
        account=Account(id=account_id, name=full_name.account_name),
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
    expected_account_name = "acc_name"
    expected_project_name = "project_name"
    project_full_name = ProjectFullName(
        account_name=expected_account_name,
        project_name=expected_project_name,
    )
    expected_project = _get_random_project(project_full_name)
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project=Mock(
            return_value=expected_project,
            spec_set=ProjectServiceClient.get_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    my_fabric = Fabric.F_MEDIUM
    pip_requirements_file_name = "req.txt"
    # when
    run_initializer = RunInitializer(
        layer_client_mock, layer_base_url=URL("https://layer.ai")
    )
    project = run_initializer.setup_project(
        project_full_name=project_full_name,
        fabric=my_fabric,
        pip_requirements_file=pip_requirements_file_name,
    )

    # then
    assert project.id == expected_project.id
    current_project_name = context.get_project_full_name()
    assert current_project_name is not None
    assert current_project_name.project_name == expected_project_name
    assert context.default_fabric() == my_fabric
    assert context.get_pip_requirements_file() == pip_requirements_file_name
    project_client_mock.get_project.assert_called_once()
    project_client_mock.create_project.assert_not_called()


def test_given_project_not_exists_when_set_up_project_creates_and_sets_global_project() -> None:
    # given
    expected_account_name = "acc_name"
    expected_project_name = "project_name"
    project_full_name = ProjectFullName(
        account_name=expected_account_name,
        project_name=expected_project_name,
    )
    expected_project = _get_random_project(project_full_name)
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project=Mock(
            return_value=None,
            spec_set=ProjectServiceClient.get_project,
        ),
        create_project=Mock(
            return_value=expected_project,
            spec_set=ProjectServiceClient.create_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    my_fabric = Fabric.F_MEDIUM
    pip_packages = ["sklearn==0.0"]

    # when
    run_initializer = RunInitializer(
        layer_client_mock,
        layer_base_url=URL("https://layer.ai"),
    )
    project = run_initializer.setup_project(
        project_full_name=project_full_name,
        fabric=my_fabric,
        pip_packages=pip_packages,
    )

    # then
    assert project.id == expected_project.id
    current_project_name = context.get_project_full_name()
    assert current_project_name is not None
    assert current_project_name.project_name == expected_project_name
    assert context.default_fabric() == my_fabric
    assert context.get_pip_packages() == pip_packages
    project_client_mock.get_project.assert_called_once()
    project_client_mock.create_project.assert_called_once()


def test_given_readme_exists_when_set_up_project_gets_and_sets_project_readme(
    tmp_path,
) -> None:
    # given
    readme_path = tmp_path / "README.md"

    expected_readme = "Sample README"
    with open(readme_path, "w") as f:
        f.write(expected_readme)

    expected_account_name = "acc_name"
    expected_project_name = "project_name"
    project_full_name = ProjectFullName(
        account_name=expected_account_name,
        project_name=expected_project_name,
    )
    expected_project = _get_random_project(project_full_name)
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project=Mock(
            return_value=expected_project,
            spec_set=ProjectServiceClient.get_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)

    # when
    run_initializer = RunInitializer(
        layer_client_mock,
        layer_base_url=URL("https://layer.ai"),
    )
    run_initializer.setup_project(
        project_full_name=project_full_name,
        project_root_path=tmp_path,
    )

    # then
    layer_client_mock.project_service_client.update_project_readme.assert_called_with(
        project_full_name=project_full_name, readme=expected_readme
    )


def test_given_readme_not_exists_when_set_up_project_gets_and_setup_project(
    tmp_path,
) -> None:
    # given
    expected_account_name = "acc_name"
    expected_project_name = "project_name"
    project_full_name = ProjectFullName(
        account_name=expected_account_name,
        project_name=expected_project_name,
    )
    expected_project = _get_random_project(project_full_name)
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project=Mock(
            return_value=expected_project,
            spec_set=ProjectServiceClient.get_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)

    # when
    run_initializer = RunInitializer(
        layer_client_mock,
        layer_base_url=URL("https://layer.ai"),
    )
    project = run_initializer.setup_project(
        project_full_name=project_full_name,
        project_root_path=tmp_path,
    )

    # then
    assert project.name == expected_project_name


def test_given_long_readme_exists_when_set_up_project_gets_and_sets_project_readme(
    tmp_path,
) -> None:
    # given
    readme_path = tmp_path / "ReAdmE.Md"

    actual_readme = "".join("." for i in range(30_000))
    expected_readme = actual_readme[:25_000]
    with open(readme_path, "w") as f:
        f.write(actual_readme)

    expected_account_name = "acc_name"
    expected_project_name = "project_name"
    project_full_name = ProjectFullName(
        account_name=expected_account_name,
        project_name=expected_project_name,
    )
    expected_project = _get_random_project(project_full_name)
    project_client_mock = MagicMock(
        set_spec=ProjectServiceClient,
        get_project=Mock(
            return_value=expected_project,
            spec_set=ProjectServiceClient.get_project,
        ),
    )
    layer_client_mock = _get_mock_layer_client(project_client_mock)
    # when
    run_initializer = RunInitializer(
        layer_client_mock,
        layer_base_url=URL("https://layer.ai"),
    )
    run_initializer.setup_project(
        project_full_name=project_full_name,
        project_root_path=tmp_path,
    )

    # then
    layer_client_mock.project_service_client.update_project_readme.assert_called_with(
        project_full_name=project_full_name, readme=expected_readme
    )
