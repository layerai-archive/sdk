import uuid
from logging import Logger
from typing import TYPE_CHECKING, List, Tuple

from layerapi.api.entity.user_log_line_pb2 import UserLogLine
from layerapi.api.ids_pb2 import RunId
from layerapi.api.service.user_logs.user_logs_api_pb2 import (
    GetPipelineRunLogsRequest,
    GetPipelineRunLogsResponse,
)
from layerapi.api.service.user_logs.user_logs_api_pb2_grpc import UserLogsAPIStub

from layer.config import ClientConfig
from layer.utils.grpc.channel import get_grpc_channel


if TYPE_CHECKING:
    pass


class UserLogsClient:
    _service: UserLogsAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._grpc_gateway_address = config.grpc_gateway_address
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @staticmethod
    def create(config: ClientConfig, logger: Logger) -> "UserLogsClient":
        client = UserLogsClient(config=config, logger=logger)
        channel = get_grpc_channel(config)
        client._service = UserLogsAPIStub(channel)  # pylint: disable=protected-access
        return client

    def get_pipeline_run_logs(
        self, run_id: uuid.UUID, continuation_token: str
    ) -> Tuple[List[UserLogLine], str]:
        response: GetPipelineRunLogsResponse = self._service.GetPipelineRunLogs(
            request=GetPipelineRunLogsRequest(
                run_id=RunId(value=str(run_id)), continuation_token=continuation_token
            )
        )
        return list(response.log_lines), response.continuation_token
