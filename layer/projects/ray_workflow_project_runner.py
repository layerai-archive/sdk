import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

import ray
from ray import workflow

from layer.clients.layer import LayerClient
from layer.config.config import Config
from layer.config.config_manager import ConfigManager
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.executables.entrypoint.common import (
    ENV_LAYER_API_TOKEN,
    ENV_LAYER_API_URL,
    ENV_LAYER_FABRIC,
)
from layer.executables.runtime import BaseFunctionRuntime
from layer.projects.execution_planner import Stage, build_plan
from layer.projects.utils import upload_executable_packages
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger()


class RayWorkflowProjectRunner:
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
            asyncio_run_in_thread(
                upload_executable_packages(
                    client, self.definitions, self.project_full_name
                )
            )
        if not ray.is_initialized():
            _metadata = [
                ("authorization", f"Bearer {self._config.credentials.access_token}")
            ]
            ray.init(address=self.ray_address, _metadata=_metadata)
        plan = build_plan(self.definitions)
        run_id = uuid.uuid4()
        workflow.run(run_stage.bind(plan.stages), workflow_id=run_id)
        run = Run(
            id=run_id,
            project_full_name=self.project_full_name,
        )
        ray.shutdown()
        return run


@ray.remote
def run_function(executable_path: Path) -> None:
    BaseFunctionRuntime.execute(executable_path=executable_path)


@ray.remote
def wait_all(*args: Any) -> None:
    pass


@ray.remote
def run_stage(stages: List[Stage], *deps: Any) -> None:
    def _get_options(layer_function: FunctionDefinition) -> Dict[str, Any]:
        config = ConfigManager().load()
        runtime_env = {
            "env_vars": {
                ENV_LAYER_API_URL: str(config.url),
                ENV_LAYER_API_TOKEN: config.credentials.access_token,
                ENV_LAYER_FABRIC: layer_function.fabric.value,
            },
        }
        if layer_function.conda_env:
            environment = layer_function.conda_env.environment
            if "name" not in environment:
                environment["name"] = "layer"
            dependencies = environment["dependencies"]
            if "pip" not in dependencies:
                dependencies.append("pip")
                dependencies.append({"pip": ["layer"]})
                environment["dependencies"] = dependencies
            else:
                pip_dict: Dict[str, Any] = next(
                    (
                        item
                        for item in dependencies
                        if isinstance(item, dict) and "pip" in item
                    ),
                    {},
                )
                if pip_dict:
                    pip_dict["pip"].append("layer")
                else:
                    pip_dict["pip"] = ["layer"]
            runtime_env["conda"] = environment
        else:
            runtime_env["pip"] = [  # type:ignore
                "layer",
                *[p for p in layer_function.pip_dependencies],
            ]
        return {
            "num_cpus": layer_function.fabric.cpu,
            "num_gpus": layer_function.fabric.gpu,
            "runtime_env": runtime_env,
        }

    if len(stages) == 0:
        return
    stage = stages.pop(0)
    function_runs = []
    for function in stage.definitions:
        function_runs.append(
            run_function.options(**_get_options(function)).bind(  # type:ignore
                function.package_download_url
            )
        )
    return workflow.continuation(run_stage.bind(stages, wait_all.bind(*function_runs)))
