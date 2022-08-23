import asyncio
import logging
import os
import urllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from urllib.parse import urlparse

import aiohttp

from layer.clients.layer import LayerClient
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition, ResourcePath
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.tracker import ResourceTransferState
from layer.tracker.progress_tracker import RunProgressTracker


def _strip_resource_root_path(path: str) -> str:
    """
    For the resource paths of the form /<organisation_id>/<project>/resources/<function>/<subdir1>/...,
    return relative resource paths. For the example above, the result would be <subdir1>/...
    :param path: resource path.
    :return: relative function resource path.
    """
    unquoted_path = urllib.parse.unquote(path)
    n = 0
    for i, c in enumerate(unquoted_path):
        if c == "/":
            n += 1
            if n == 5:
                return unquoted_path[(i + 1) :]
    return unquoted_path


class ResourceManager:
    def __init__(self, client: LayerClient):
        self._client = client
        self._log = logging.getLogger(__name__)

    async def _upload_resource(
        self,
        project_full_name: ProjectFullName,
        function_name: str,
        resource_path: ResourcePath,
        file_path: str,
        session: aiohttp.ClientSession,
        state: ResourceTransferState,
    ) -> None:
        """
        Upload single local resource file.
        """
        self._log.debug(
            f"Upload resource file {file_path} of function {function_name} to {resource_path.path}"
        )
        resource_uri = self._client.data_catalog.get_resource_paths(
            project_full_name=project_full_name,
            function_name=function_name,
            path=resource_path.path,
        )[0]
        with open(file_path, mode="rb") as f:
            await session.put(
                resource_uri,
                data=f,
                timeout=aiohttp.ClientTimeout(total=None),
                trace_request_ctx={"state": state},
            )
        state.increment_num_transferred_files(1)

    def _update_resource_paths_index(
        self, project_full_name: ProjectFullName, functions: List[FunctionDefinition]
    ) -> None:
        for function in functions:
            self._client.data_catalog.update_resource_paths_index(
                project_full_name=project_full_name,
                function_name=function.func_name,
                paths=[
                    local_path
                    for resource_path in function.resource_paths
                    for local_path in resource_path.local_relative_paths()
                ],
            )

    @staticmethod
    async def _on_request_chunk_sent(
        session: aiohttp.ClientSession,
        ctx: SimpleNamespace,
        params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        state: ResourceTransferState = ctx.trace_request_ctx["state"]
        state.increment_transferred_resource_size_bytes(len(params.chunk))

    async def _upload_resources(
        self,
        project_full_name: ProjectFullName,
        functions: List[FunctionDefinition],
        tracker: RunProgressTracker,
    ) -> None:
        """
        Collect and upload local files as resources for all functions decorated with `@resource`.
        """
        self._update_resource_paths_index(project_full_name, functions)
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_chunk_sent.append(self._on_request_chunk_sent)
        async with aiohttp.ClientSession(
            raise_for_status=True, trace_configs=[trace_config]
        ) as session:
            upload_tasks = [
                self._upload_resources_for_function(
                    project_full_name, function, tracker, session
                )
                for function in functions
            ]
            await asyncio.gather(*upload_tasks)

    async def _upload_resources_for_function(
        self,
        project_full_name: ProjectFullName,
        function: FunctionDefinition,
        tracker: RunProgressTracker,
        session: aiohttp.ClientSession,
    ) -> None:
        state = ResourceTransferState()
        upload_tasks = []
        total_num_files = 0
        total_files_size_bytes = 0
        for resource_path in function.resource_paths:
            for local_path in resource_path.local_relative_paths():
                total_num_files += 1
                total_files_size_bytes += os.path.getsize(os.path.abspath(local_path))
                upload_task = self._upload_resource(
                    project_full_name=project_full_name,
                    function_name=function.func_name,
                    resource_path=ResourcePath(path=local_path),
                    file_path=os.path.abspath(local_path),
                    session=session,
                    state=state,
                )
                upload_tasks.append(upload_task)
        state.total_num_files = total_num_files
        state.total_resource_size_bytes = total_files_size_bytes

        asset_type = function.asset_type
        asset_name = function.asset_name
        if asset_type == AssetType.DATASET:
            tracker.mark_dataset_resources_uploading(asset_name, state)
        elif asset_type == AssetType.MODEL:
            tracker.mark_model_resources_uploading(asset_name, state)
        await asyncio.gather(*upload_tasks)
        if asset_type == AssetType.DATASET:
            tracker.mark_dataset_resources_uploaded(asset_name)
        elif asset_type == AssetType.MODEL:
            tracker.mark_model_resources_uploaded(asset_name)

    def wait_resource_upload(
        self,
        project_full_name: ProjectFullName,
        functions: List[FunctionDefinition],
        tracker: RunProgressTracker,
    ) -> None:
        """
        Collect and upload local files as resources for all functions decorated with `@resources`.
        """
        self._log.debug("Uploading resources")
        # run in a separate thread to make sure no other loop is running
        with ThreadPoolExecutor(max_workers=1) as executor:
            upload = executor.submit(
                lambda: asyncio.run(
                    self._upload_resources(project_full_name, functions, tracker)
                )
            )
            upload.result()

    async def _download_resource(
        self, download_url: str, target_dir: str, session: aiohttp.ClientSession
    ) -> Any:
        download_url_path = urlparse(download_url).path
        relative_resource_part = _strip_resource_root_path(download_url_path)
        resource_dir, resource_file = os.path.split(relative_resource_part)
        download_dir = os.path.join(target_dir, resource_dir)
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        download_path = os.path.join(download_dir, resource_file)
        async with session.get(
            download_url, timeout=aiohttp.ClientTimeout(total=None)
        ) as response:
            with open(download_path, mode="wb") as f:
                bytes_total = 0
                async for data in response.content.iter_chunked(8192):
                    bytes_total += f.write(data)
        self._log.info(
            f"Downloaded {download_url_path} to {download_path}, bytes total: {bytes_total}"
        )

    async def _download_resources(
        self, project_full_name: ProjectFullName, function_name: str, target_dir: str
    ) -> Any:
        resource_paths = self._client.data_catalog.get_resource_paths(
            project_full_name, function_name
        )
        download_root = os.getcwd() if target_dir == "" else target_dir
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            download_tasks = [
                self._download_resource(download_url, download_root, session)
                for download_url in resource_paths
            ]
            await asyncio.gather(*download_tasks)

    def wait_resource_download(
        self,
        project_full_name: ProjectFullName,
        function_name: str,
        target_dir: str = "",
    ) -> None:
        """
        Download resource files for the function.
        """
        # run in a separate thread to make sure no other loop is running
        with ThreadPoolExecutor(max_workers=1) as executor:
            download = executor.submit(
                lambda: asyncio.run(
                    self._download_resources(
                        project_full_name, function_name, target_dir
                    )
                )
            )
            download.result()
