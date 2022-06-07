import hashlib
import uuid
from typing import List, Optional
from uuid import UUID

from layer.clients.layer import LayerClient
from layer.contracts.accounts import Account
from layer.contracts.projects import Project, ProjectFullName
from layer.contracts.runs import FunctionDefinition
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import current_account_name, current_project_name


def verify_current_project_exists_and_retrieve_project_id(client: LayerClient) -> UUID:
    parent_account = get_specified_account_or_default_to_personal(
        client, current_account_name()
    )
    return verify_project_exists_and_retrieve_project_id(
        client, parent_account, get_current_project_name()
    )


def verify_account_project_exists_and_retrieve_project_id(
    client: LayerClient,
    project_full_name: ProjectFullName,
) -> UUID:
    account = get_specified_account_or_default_to_personal(
        client, project_full_name.account_name
    )
    return verify_project_exists_and_retrieve_project_id(
        client, account, project_full_name.project_name
    )


def verify_project_exists_and_retrieve_project_id(
    client: LayerClient, parent_account: Account, project_name: str
) -> UUID:
    project_uuid = client.project_service_client.get_project_id(
        parent_account, project_name
    )
    if not project_uuid:
        raise ProjectInitializationException(
            f"Project with the name {project_name} does not exist in account {parent_account.name}"
        )
    return project_uuid


def get_remote_project(client: LayerClient, project_id: uuid.UUID) -> Project:
    project = client.project_service_client.get_project(project_id)
    assert project is not None
    return project


def get_or_create_remote_project(
    client: LayerClient, parent_account: Account, project_name: str
) -> Project:
    project_id = client.project_service_client.get_project_id(
        parent_account, project_name
    )
    if project_id is None:
        project_id = client.project_service_client.create_project(
            parent_account,
            project_name,
        )
    assert project_id is not None
    return Project(
        name=project_name,
        account=parent_account,
        _id=project_id,
    )


def get_specified_account_or_default_to_personal(
    layer_client: LayerClient, account_name: Optional[str]
) -> Account:
    if account_name is None:
        return layer_client.account.get_my_account()
    else:
        return layer_client.account.get_account_by_name(account_name)


def get_current_project_name() -> str:
    current_project_name_ = current_project_name()
    if not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')`"
        )
    return current_project_name_


def get_current_project_full_name() -> ProjectFullName:
    current_project_name_ = current_project_name()
    if not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')` "
            "or `layer.init('account-name/project-name')`"
        )
    current_account_name_ = current_account_name()
    return ProjectFullName(
        project_name=current_project_name_,
        account_name=current_account_name_,
    )


def calculate_hash_by_definitions(definitions: List[FunctionDefinition]) -> str:
    files_hash = hashlib.sha256()

    for definition in definitions:
        files_hash.update(definition.get_pickled_function())

    return files_hash.hexdigest()
