from contextlib import contextmanager
from logging import Logger
from typing import Iterator

from layerapi.api.service.executor.executor_api_pb2 import GetFunctionUploadPathRequest
from layerapi.api.service.executor.executor_api_pb2_grpc import ExecutorAPIStub

from layer.config import ClientConfig
from layer.contracts.project_full_name import ProjectFullName
from layer.utils.grpc import create_grpc_channel


class ExecutorClient:
    _service: ExecutorAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["ExecutorClient"]:
        with create_grpc_channel(
            self._config.grpc_gateway_address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = ExecutorAPIStub(channel=channel)
            yield self

    def get_upload_path(
        self,
        project_full_name: ProjectFullName,
        function_name: str,
    ) -> str:
        request = GetFunctionUploadPathRequest(
            project_full_name=project_full_name.path,
            function_name=function_name,
        )
        return self._service.GetFunctionUploadPath(request=request).function_upload_path
