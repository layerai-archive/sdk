from layerapi.api.service.account.user_api_pb2 import GetGuestAuthTokenRequest
from layerapi.api.service.account.user_api_pb2_grpc import UserAPIStub

from layer.config import LogsConfig
from layer.utils.grpc import create_grpc_channel


def get_guest_auth_token(url: str) -> str:
    with create_grpc_channel(
        url, "", logs_file_path=LogsConfig().logs_file_path
    ) as channel:
        svc = UserAPIStub(channel=channel)
        auth_token = svc.GetGuestAuthToken(GetGuestAuthTokenRequest())
        return auth_token.token
