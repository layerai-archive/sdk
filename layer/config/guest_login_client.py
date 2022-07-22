from collections import namedtuple

from layerapi.api.service.account.user_api_pb2 import GetGuestAuthTokenRequest
from layerapi.api.service.account.user_api_pb2_grpc import UserAPIStub

from layer.config import LogsConfig
from layer.utils.grpc.channel import get_grpc_channel


_ChannelConfig = namedtuple(
    "_ChannelConfig",
    ("grpc_gateway_address", "access_token", "logs_file_path", "grpc_do_verify_ssl"),
)


def _get_channel_config(url: str) -> _ChannelConfig:
    return _ChannelConfig(
        grpc_gateway_address=url,
        access_token="",
        logs_file_path=LogsConfig().logs_file_path,
        grpc_do_verify_ssl=True,
    )


def get_guest_auth_token(url: str) -> str:
    config = _get_channel_config(url)
    channel = get_grpc_channel(config)
    user_service = UserAPIStub(channel)
    auth_token = user_service.GetGuestAuthToken(GetGuestAuthTokenRequest())
    return auth_token.token
