import logging
from typing import Any, Dict, Tuple

from layer.context import Context
from layer.contracts.datasets import DatasetBuild
from layer.contracts.runs import TaskType
from layer.utils.runtime_utils import check_and_convert_to_df

from .common import FunctionRunner


logger = logging.getLogger(__name__)


class DatasetRunner(FunctionRunner):
    build: DatasetBuild

    def _create_asset(self) -> Tuple[str, Dict[str, Any]]:
        self.client.data_catalog.add_dataset(
            asset_path=self.definition.asset_path,
            project_id=self.project_id,
            description=self.definition.description,
            fabric=self.fabric.value,
            func_source=self.definition.func_source,
            entrypoint=self.definition.entrypoint,
            environment=self.definition.environment,
        )
        dataset_build_id = self.client.data_catalog.initiate_build(
            self.project_id,
            self.definition.asset_name,
            self.fabric.value,
        )

        self.build = self.client.data_catalog.get_build_by_id(dataset_build_id)

        if self.run_id:
            self.client.flow_manager.update_run_metadata(
                run_id=self.run_id,
                task_id=self.definition.asset_path.path(),
                task_type=TaskType.DATASET_BUILD,
                key="build-id",
                value=str(self.build.id),
            )

        return self.build.tag, {"dataset_build": self.build}

    def _mark_start(self) -> None:
        pass

    def _mark_success(self) -> str:
        build = self.client.data_catalog.get_build_by_id(self.build.id)
        return build.tag

    def _mark_failure(self, failure_exc: Exception) -> str:
        build = self.client.data_catalog.complete_build(
            self.build.id,
            self.definition.asset_name,
            self.definition.uri,
            error=failure_exc,
        )
        return build.tag

    def _transform_output(self, output: Any) -> Any:
        return check_and_convert_to_df(output)

    def _save_artifact(self, ctx: Context, artifact: Any) -> None:
        ctx.save_dataset(artifact)
