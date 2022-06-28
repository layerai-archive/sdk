from contextlib import contextmanager
from logging import Logger
from typing import Iterator, List, Tuple

from layerapi.api.entity.history_event_pb2 import HistoryEvent
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.entity.run_metadata_pb2 import RunMetadata
from layerapi.api.entity.run_pb2 import Run
from layerapi.api.ids_pb2 import RunId
from layerapi.api.service.flowmanager.flow_manager_api_pb2 import (
    GetRunByIdRequest,
    GetRunHistoryAndMetadataRequest,
    StartRunV2Request,
)
from layerapi.api.service.flowmanager.flow_manager_api_pb2_grpc import (
    FlowManagerAPIStub,
)
from layerapi.api.value.sha256_pb2 import Sha256

from layer.config import ClientConfig
from layer.contracts.project_full_name import ProjectFullName
from layer.utils.grpc import create_grpc_channel


class FlowManagerClient:
    _service: FlowManagerAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.flow_manager
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["FlowManagerClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = FlowManagerAPIStub(channel=channel)
            yield self

    def start_run(
        self,
        project_full_name: ProjectFullName,
        execution_plan: ExecutionPlan,
        project_files_hash: str,
        user_command: str,
    ) -> RunId:
        response = self._service.StartRunV2(
            request=StartRunV2Request(
                project_full_name=project_full_name.path,
                plan=execution_plan,
                project_files_hash=Sha256(value=project_files_hash),
                user_command=user_command,
            )
        )
        return response.run_id

    def get_run(self, run_id: RunId) -> Run:
        response = self._service.GetRunById(GetRunByIdRequest(run_id=run_id))
        return response.run

    def get_run_status_history_and_metadata(
        self, run_id: RunId
    ) -> Tuple[List[HistoryEvent], RunMetadata]:
        response = self._service.GetRunHistoryAndMetadata(
            GetRunHistoryAndMetadataRequest(run_id=run_id)
        )
        return list(response.events), response.run_metadata
