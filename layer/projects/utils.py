import hashlib
from typing import List
from uuid import UUID

from layer import current_project_name
from layer.clients.layer import LayerClient
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.contracts.runs import FunctionDefinition
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import current_account_name


def verify_project_exists_and_retrieve_project_id(
    client: LayerClient, project_full_name: ProjectFullName
) -> UUID:
    project = client.project_service_client.get_project(project_full_name)
    if not project:
        raise ProjectInitializationException(
            f"Project '{project_full_name.path}' does not exist."
        )
    return project.id


def get_or_create_remote_project(
    client: LayerClient, project_full_name: ProjectFullName
) -> Project:
    project = client.project_service_client.get_project(project_full_name)
    if project is not None:
        return project

    return client.project_service_client.create_project(
        project_full_name,
    )


def get_current_project_name() -> str:
    current_project_name_ = current_project_name()
    if not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')`"
        )
    return current_project_name_


def get_current_project_full_name() -> ProjectFullName:
    current_account_name_ = current_account_name()
    current_project_name_ = current_project_name()
    if not current_account_name_ or not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')`"
        )
    return ProjectFullName(
        account_name=current_account_name_,
        project_name=current_project_name_,
    )


def calculate_hash_by_definitions(definitions: List[FunctionDefinition]) -> str:
    files_hash = hashlib.sha256()

    for definition in definitions:
        files_hash.update(definition.get_pickled_function())

    return files_hash.hexdigest()
