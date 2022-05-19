from uuid import UUID

from layer import current_project_name
from layer.clients.layer import LayerClient
from layer.contracts.accounts import Account
from layer.contracts.projects import Project
from layer.exceptions.exceptions import ProjectInitializationException


def verify_project_exists_and_retrieve_project_id(
    client: LayerClient, project_name: str
) -> UUID:
    project_uuid = client.project_service_client.get_project_id_and_org_id(
        project_name
    ).project_id
    if not project_uuid:
        raise ProjectInitializationException(
            f"Project with the name {project_name} does not exist."
        )
    return project_uuid


def get_or_create_remote_project(client: LayerClient, project: Project) -> Project:
    project_id_with_org_id = client.project_service_client.get_project_id_and_org_id(
        project.name
    )
    if project_id_with_org_id.project_id is None:
        project_id_with_org_id = client.project_service_client.create_project(
            project.name
        )
    assert project_id_with_org_id.project_id is not None
    assert project_id_with_org_id.account_id is not None
    account_name = client.account.get_account_name_by_id(
        project_id_with_org_id.account_id
    )
    account = Account(id=project_id_with_org_id.account_id, name=account_name)
    return project.with_id(project_id_with_org_id.project_id).with_account(
        account=account
    )


def get_current_project_name() -> str:
    current_project_name_ = current_project_name()
    if not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')`"
        )
    return current_project_name_
