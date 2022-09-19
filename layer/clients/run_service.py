from typing import Optional
from uuid import UUID

from layerapi.api.service.logged_data.run_api_pb2 import (
    CreateRunRequest,
    FinishRunRequest,
    GetRunByRunIdRequest,
)
from layerapi.api.service.logged_data.run_api_pb2_grpc import RunAPIStub

from layer.config import ClientConfig
from layer.contracts.runs import Run, RunStatus
from layer.utils.grpc.channel import get_grpc_channel

from .protomappers import projects as project_proto_mapper
from .protomappers import runs as run_proto_mapper


class RunServiceClient:
    _service: RunAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "RunServiceClient":
        client = RunServiceClient()
        channel = get_grpc_channel(config)
        client._service = RunAPIStub(channel)  # pylint: disable=protected-access
        return client

    def get_run_by_id(self, run_id: UUID) -> Run:
        request = GetRunByRunIdRequest(
            run_id=run_proto_mapper.to_run_id(run_id),
        )
        run_pb = self._service.GetRunByRunId(request=request).run
        return run_proto_mapper.from_run(run_pb)

    def get_run_by_index(self, project_id: UUID, run_index: int) -> Run:
        raise NotImplementedError()

    def create_run(self, project_id: UUID, run_name: Optional[str] = None) -> Run:
        request = CreateRunRequest(
            project_id=project_proto_mapper.to_project_id(project_id),
        )
        if run_name:
            request.name = run_name
        run_pb = self._service.CreateRun(request=request).run
        return run_proto_mapper.from_run(run_pb)

    def finish_run(self, run_id: UUID, status: RunStatus) -> Run:
        request = FinishRunRequest(
            run_id=run_proto_mapper.to_run_id(run_id),
            status=run_proto_mapper.to_run_status(status),
        )
        run_pb = self._service.FinishRun(request=request).run
        return run_proto_mapper.from_run(run_pb)
