import asyncio
import hashlib
import re
from typing import List

import aiohttp

from layer.clients.layer import LayerClient
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName


_PROJECT_NAME_PATTERN = re.compile(r"^(([a-zA-Z0-9_-]+)\/)?([a-zA-Z0-9_-]+)$")


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
        files_hash.update(definition.source_code_digest.encode())

    return files_hash.hexdigest()


async def upload_executable_packages(
    client: LayerClient,
    definitions: List[FunctionDefinition],
    project_full_name: ProjectFullName,
) -> None:
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        upload_tasks = [
            _upload_executable_package(client, project_full_name, definition, session)
            for definition in definitions
        ]
        await asyncio.gather(*upload_tasks)


async def _upload_executable_package(
    client: LayerClient,
    project_full_name: ProjectFullName,
    function: FunctionDefinition,
    session: aiohttp.ClientSession,
) -> None:
    function.package()
    function_name = (
        f"{function.asset_path.asset_type.value}/{function.asset_path.asset_name}"
    )
    with open(function.executable_path, "rb") as package_file:
        presigned_url = client.executor_service_client.get_function_upload_path(
            project_full_name=project_full_name,
            function_name=function_name,
        )
        await session.put(
            presigned_url,
            data=package_file,
            timeout=aiohttp.ClientTimeout(total=None),
        )
        download_url = client.executor_service_client.get_function_download_path(
            project_full_name=project_full_name,
            function_name=function_name,
        )
        function.set_package_download_url(download_url)
