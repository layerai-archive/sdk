import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, List

import aiohttp
import ray
from layerapi.api.ids_pb2 import RunId
from ray import workflow

from layer.clients.layer import LayerClient
from layer.config.config import Config
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.executables.runtime import BaseFunctionRuntime
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger()


class RayProjectRunner:
    def __init__(
        self,
        config: Config,
        project_full_name: ProjectFullName,
        functions: List[Any],
        ray_address: str,
    ) -> None:
        self._config = config
        self.project_full_name = project_full_name
        self.definitions: List[FunctionDefinition] = [
            f.get_definition_with_bound_arguments() for f in functions
        ]
        self.ray_address = ray_address

    def run(self) -> Run:
        with LayerClient(self._config.client, logger).init() as client:
            asyncio_run_in_thread(self._upload_executable_packages(client))
        from layer.projects.execution_planner import build_execution_plan

        plan = build_execution_plan(self.definitions)
        serialized_ops = [op.SerializeToString() for op in plan.operations]
        run_id = str(uuid.uuid4())
        workflow.run(run_stage.bind(serialized_ops), workflow_id=run_id)
        run = Run(
            id=RunId(value=run_id),
            project_full_name=self.project_full_name,
        )
        return run

    async def _upload_executable_packages(self, client: LayerClient) -> None:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            upload_tasks = [
                self._upload_executable_package(client, definition, session)
                for definition in self.definitions
            ]
            await asyncio.gather(*upload_tasks)

    async def _upload_executable_package(
        self,
        client: LayerClient,
        function: FunctionDefinition,
        session: aiohttp.ClientSession,
    ) -> None:
        function.package()
        function_name = (
            f"{function.asset_path.asset_type.value}/{function.asset_path.asset_name}"
        )
        with open(function.executable_path, "rb") as package_file:
            presigned_url = client.executor_service_client.get_function_upload_path(
                project_full_name=self.project_full_name,
                function_name=function_name,
            )
            await session.put(
                presigned_url,
                data=package_file,
                timeout=aiohttp.ClientTimeout(total=None),
            )
            download_url = client.executor_service_client.get_function_download_path(
                project_full_name=self.project_full_name,
                function_name=function_name,
            )
            function.set_package_download_url(download_url)


@ray.remote
def run_function(executable_path: Path) -> None:
    BaseFunctionRuntime.execute(executable_path=executable_path)


@ray.remote
def wait_all(*args) -> None:
    pass


@ray.remote
def run_stage(operations: List[str], *deps) -> None:
    from layerapi.api.entity.operations_pb2 import Operation

    if len(operations) == 0:
        return
    serialized_op = operations.pop(0)
    operation = Operation()
    operation.ParseFromString(serialized_op)
    function_runs = []
    if operation.HasField("sequential"):
        function_runs.append(
            run_function.bind(
                operation.sequential.function_execution.executable_package_url
            )
        )
    else:
        for function in operation.parallel.function_execution:
            function_runs.append(run_function.bind(function.executable_package_url))
    return workflow.continuation(
        run_stage.bind(operations, wait_all.bind(*function_runs))
    )
