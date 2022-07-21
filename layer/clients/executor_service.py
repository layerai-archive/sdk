from logging import Logger

from layerapi.api.service.executor.executor_api_pb2 import GetFunctionUploadPathRequest
from layerapi.api.service.executor.executor_api_pb2_grpc import ExecutorAPIStub

from layer.config import ClientConfig
from layer.contracts.project_full_name import ProjectFullName
from layer.utils.grpc.channel import get_grpc_channel


class ExecutorClient:
    _service: ExecutorAPIStub

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
    def create(config: ClientConfig, logger: Logger) -> "ExecutorClient":
        client = ExecutorClient(config=config, logger=logger)
        channel = get_grpc_channel(config)
        client._service = ExecutorAPIStub(channel)  # pylint: disable=protected-access
        return client

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
