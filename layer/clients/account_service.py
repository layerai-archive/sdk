import uuid
from contextlib import contextmanager
from logging import Logger
from typing import Iterator

from layerapi.api.ids_pb2 import AccountId
from layerapi.api.service.account.account_api_pb2 import (
    GetAccountViewByIdRequest,
    GetAccountViewByIdResponse,
)
from layerapi.api.service.account.account_api_pb2_grpc import AccountAPIStub
from layerapi.api.service.account.user_api_pb2 import GetMyOrganizationRequest
from layerapi.api.service.account.user_api_pb2_grpc import UserAPIStub

from layer.config import ClientConfig
from layer.contracts.accounts import Account
from layer.utils.grpc import create_grpc_channel


class AccountServiceClient:
    _service: UserAPIStub
    _account_api: AccountAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.account_service
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["AccountServiceClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = UserAPIStub(channel=channel)
            self._account_api = AccountAPIStub(channel=channel)
            yield self

    def get_account_name_by_id(self, account_id: uuid.UUID) -> str:
        get_my_org_resp: GetAccountViewByIdResponse = (
            self._account_api.GetAccountViewById(
                GetAccountViewByIdRequest(id=AccountId(value=str(account_id))),
            )
        )
        return get_my_org_resp.account_view.name

    def get_my_account(self) -> Account:
        # TODO LAY-2692
        org = self._service.GetMyOrganization(
            GetMyOrganizationRequest(),
        ).organization
        return Account(
            id=uuid.UUID(org.id.value),
            name=org.name,
        )
