import hashlib
import re
from typing import List
from uuid import UUID

from layer.clients.layer import LayerClient
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import current_project_full_name


_PROJECT_NAME_PATTERN = re.compile(r"^(([a-zA-Z0-9_-]+)\/)?([a-zA-Z0-9_-]+)$")


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


def get_current_project_full_name() -> ProjectFullName:
    project_full_name = current_project_full_name()
    if not project_full_name:
        raise ProjectInitializationException(
            "Please specify the current project name globally with"
            " 'layer.init(\"account-name/project-name\")' or 'layer.init(\"project-name\")'"
        )
    return project_full_name


def validate_project_name(project_name: str) -> None:
    result = _PROJECT_NAME_PATTERN.search(project_name)
    if not result:
        raise ValueError(
            "Invalid project or account name. Please provide project name in the format"
            ' "account-name/project-name" or "project-name". Names can only'
            " contain letters a-z A-Z, numbers 0-9, hyphens (-) and underscores (_)."
        )


def calculate_hash_by_definitions(definitions: List[FunctionDefinition]) -> str:
    files_hash = hashlib.sha256()

    for definition in definitions:
        files_hash.update(definition.get_pickled_function())

    return files_hash.hexdigest()
