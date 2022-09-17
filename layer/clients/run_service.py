from typing import Optional

from layerapi import api

from layer.config import ClientConfig
from layer.contracts import ids
from layer.contracts.runs import Run, RunStatus
from layer.utils.grpc.channel import get_grpc_channel

from .protomappers import projects as project_proto_mapper
from .protomappers import runs as run_proto_mapper


class RunServiceClient:
    _service: api.RunAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "RunServiceClient":
        client = RunServiceClient()
        channel = get_grpc_channel(config)
        client._service = api.RunAPIStub(channel)  # pylint: disable=protected-access
        return client

    async def get_run_by_id(self, run_id: ids.RunId) -> Run:
        response = await self._service.get_run_by_run_id(
            run_id=run_proto_mapper.to_run_id(run_id),
        )
        return run_proto_mapper.from_run(response.run)

    async def get_run_by_index(self, project_id: ids.ProjectId, run_index: int) -> Run:
        raise NotImplementedError()

    async def create_run(
        self, project_id: ids.ProjectId, run_name: Optional[str] = None
    ) -> Run:
        response = await self._service.create_run(
            project_id=project_proto_mapper.to_project_id(project_id),
            name=run_name,  # TODO(volkan) make this optional
        )
        return run_proto_mapper.from_run(response.run)

    async def finish_run(self, run_id: ids.RunId, status: RunStatus) -> Run:
        response = await self._service.finish_run(
            run_id=run_proto_mapper.to_run_id(run_id),
            status=run_proto_mapper.to_run_status(status),
        )
        return run_proto_mapper.from_run(response.run)
