from typing import List, Mapping, Tuple

from layerapi import api

from layer.config import ClientConfig
from layer.contracts import ids
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import TaskType
from layer.utils.grpc.channel import get_grpc_channel

from .protomappers import runs as runs_proto_mapper


class FlowManagerClient:
    _service: api.FlowManagerAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "FlowManagerClient":
        client = FlowManagerClient()
        channel = get_grpc_channel(config)
        client._service = api.FlowManagerAPIStub(  # pylint: disable=protected-access
            channel
        )
        return client

    async def start_run(
        self,
        project_full_name: ProjectFullName,
        execution_plan: api.ExecutionPlan,
        project_files_hash: str,
        user_command: str,
        env_variables: Mapping[str, str],
    ) -> ids.RunId:
        response = await self._service.start_run_v2(
            project_full_name=project_full_name.path,
            plan=execution_plan,
            project_files_hash=api.Sha256(value=project_files_hash),
            user_command=user_command,
            env_variables=env_variables,
        )
        return runs_proto_mapper.from_run_id(response.run_id)

    async def get_run_status_history_and_metadata(
        self, run_id: ids.RunId
    ) -> Tuple[List[api.HistoryEvent], api.RunMetadata]:
        response = await self._service.get_run_history_and_metadata(
            run_id=runs_proto_mapper.to_run_id(run_id),
        )
        return list(response.events), response.run_metadata

    async def update_run_metadata(
        self,
        run_id: ids.RunId,
        task_id: str,
        task_type: TaskType,
        key: str,
        value: str,
    ) -> None:
        run_metadata = api.RunMetadata(
            run_id=runs_proto_mapper.to_run_id(run_id),
            entries=[
                api.RunMetadataEntry(
                    task_id=task_id,
                    task_type=runs_proto_mapper.TASK_TYPE_TO_PROTO_MAP[task_type],
                    key=key,
                    value=value,
                )
            ],
        )
        await self._service.update_run_metadata(run_metadata=run_metadata)
