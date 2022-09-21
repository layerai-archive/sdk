import uuid
from datetime import timedelta
from typing import Tuple

from google.protobuf.duration_pb2 import Duration
from layerapi.api.ids_pb2 import AccountId
from layerapi.api.service.executor.executor_api_pb2 import (
    GetFunctionDownloadPathRequest,
    GetFunctionUploadPathRequest,
    GetOrCreateClusterRequest,
)
from layerapi.api.service.executor.executor_api_pb2_grpc import ExecutorAPIStub
from layerapi.api.value.language_version_pb2 import LanguageVersion

from layer.config import ClientConfig
from layer.contracts.project_full_name import ProjectFullName
from layer.utils.grpc.channel import get_grpc_channel


class ExecutorClient:
    _service: ExecutorAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "ExecutorClient":
        client = ExecutorClient()
        channel = get_grpc_channel(config)
        client._service = ExecutorAPIStub(channel)  # pylint: disable=protected-access
        return client

    def get_function_upload_path(
        self,
        project_full_name: ProjectFullName,
        function_name: str,
    ) -> str:
        request = GetFunctionUploadPathRequest(
            project_full_name=project_full_name.path,
            function_name=function_name,
        )
        return self._service.GetFunctionUploadPath(request=request).function_upload_path

    def get_function_download_path(
        self,
        project_full_name: ProjectFullName,
        function_name: str,
    ) -> str:
        request = GetFunctionDownloadPathRequest(
            project_full_name=project_full_name.path,
            function_name=function_name,
        )
        return self._service.GetFunctionDownloadPath(
            request=request
        ).function_download_path

    def get_or_create_cluster(
        self, account_id: uuid.UUID, language_version: Tuple[int, int, int]
    ) -> str:
        idle_scale_down_duration_td = timedelta(minutes=10)
        idle_shut_down_duration_td = timedelta(hours=1)
        idle_scale_down_duration = Duration()
        idle_scale_down_duration.FromTimedelta(td=idle_scale_down_duration_td)
        idle_shut_down_duration = Duration()
        idle_shut_down_duration.FromTimedelta(td=idle_shut_down_duration_td)
        # TODO(emin): add account_id and language_version as parameter to below
        request = GetOrCreateClusterRequest(
            idle_scale_down_duration=idle_scale_down_duration,
            idle_shut_down_duration=idle_shut_down_duration,
            max_number_of_workers=10,  # TODO(emin): remove this parameter
            worker_fabric="f-small",  # TODO(emin): remove this parameter as it should be defined by the function executable for each
            account_id=AccountId(value=str(account_id)),
            language_version=LanguageVersion(
                major=language_version[0],
                minor=language_version[1],
                micro=language_version[2],
            ),
        )
        return self._service.GetOrCreateCluster(request=request).cluster_url
