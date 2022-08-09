import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import aiohttp
from layerapi.api.entity.operations_pb2 import (
    ExecutionPlan,
    FunctionExecutionOperation,
    Operation,
    SequentialOperation,
)
from layerapi.api.entity.task_pb2 import Task
from layerapi.api.ids_pb2 import RunId

from layer.clients.executor_service import ExecutorClient
from layer.clients.flow_manager import FlowManagerClient
from layer.config.config_manager import ConfigManager
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.executables.function import DatasetOutput, Function, ModelOutput
from layer.projects.utils import get_current_project_full_name


def remote_run(funcs: Sequence[Callable[..., Any]]) -> Run:
    job = _Job.from_functions(funcs)
    project = get_current_project_full_name()

    client_config = ConfigManager().load().client
    executor_client = ExecutorClient.create(client_config)
    flow_manager_client = FlowManagerClient.create(client_config)
    remote_runner = _RemoteRunner(executor_client, flow_manager_client, project)

    run_id = remote_runner.submit_job(job)

    return Run(run_id, project)


@dataclass(frozen=True)
class _Job:
    functions: Sequence[Function]

    @staticmethod
    def from_functions(functions: Sequence[Callable[..., Any]]) -> "_Job":
        return _Job(functions=tuple(Function.from_decorated(f) for f in functions))


class _RemoteRunner:
    def __init__(
        self,
        executor_client: ExecutorClient,
        flow_manager_client: FlowManagerClient,
        project: ProjectFullName,
    ) -> None:
        self._flow_manager_client = flow_manager_client
        self._executor_client = executor_client
        self._project = project

    def submit_job(self, job: _Job) -> RunId:
        # submit job in a separate thread pool, to get own asyncio event loop
        with ThreadPoolExecutor(max_workers=1) as executor:

            def asyncio_run() -> RunId:
                return asyncio.run(self._submit_job(job))

            run_id = executor.submit(asyncio_run)
            return run_id.result(timeout=None)

    async def _submit_job(self, job: _Job) -> RunId:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            upload_tasks = (self._upload_function(session, f) for f in job.functions)
            await asyncio.gather(*upload_tasks)

        # execute each function sequentially
        execution_plan = self._sequential_execution_plan(job)

        run_id = self._flow_manager_client.start_run(
            project_full_name=self._project,
            execution_plan=execution_plan,
            project_files_hash="73475cb40a568e8da8a045ced110137e159f890ac4da883b6b17dc651b3a8049",  # TODO: check if checksum is used
            user_command="",
        )

        return run_id

    async def _upload_function(
        self, session: aiohttp.ClientSession, function: Function
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = function.package(output_dir=Path(temp_dir))
            package_uri = self._executor_client.get_function_upload_path(
                self._project, function.name
            )
            with open(package_path, mode="rb") as f:
                async with session.put(
                    package_uri, data=f, timeout=aiohttp.ClientTimeout(total=None)
                ):
                    pass

    def _sequential_execution_plan(self, job: _Job) -> ExecutionPlan:
        operations = tuple(
            Operation(
                sequential=SequentialOperation(
                    function_execution=self._function_to_execution_operation(f)
                )
            )
            for f in job.functions
        )

        return ExecutionPlan(operations=operations)

    def _function_to_execution_operation(
        self, function: Function
    ) -> FunctionExecutionOperation:
        task_type = Task.Type.TYPE_INVALID
        if isinstance(function.output, DatasetOutput):
            task_type = Task.Type.TYPE_DATASET_BUILD
        elif isinstance(function.output, ModelOutput):
            task_type = Task.Type.TYPE_MODEL_TRAIN

        return FunctionExecutionOperation(
            task_type=task_type,
            asset_name=function.output.name,
            executable_package_url=self._executor_client.get_function_download_path(
                self._project, function.name
            ),
            fabric=function.fabric.value,
        )
