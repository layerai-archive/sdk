import uuid

from layerapi.api.ids_pb2 import AccountId
from layerapi.api.service.account.account_api_pb2 import (
    GetAccountViewByIdRequest,
    GetAccountViewByIdResponse,
    GetMyAccountViewRequest,
)
from layerapi.api.service.account.account_api_pb2_grpc import AccountAPIStub

from layer.config import ClientConfig
from layer.contracts.accounts import Account
from layer.utils.grpc.channel import get_grpc_channel


class AccountServiceClient:
    _account_api: AccountAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "AccountServiceClient":
        client = AccountServiceClient()
        channel = get_grpc_channel(config)
        client._account_api = AccountAPIStub(  # pylint: disable=protected-access
            channel
        )
        return client

    def get_account_name_by_id(self, account_id: uuid.UUID) -> str:
        get_my_org_resp: GetAccountViewByIdResponse = (
            self._account_api.GetAccountViewById(
                GetAccountViewByIdRequest(id=AccountId(value=str(account_id))),
            )
        )
        return get_my_org_resp.account_view.name

    def get_my_account(self) -> Account:
        account_view = self._account_api.GetMyAccountView(
            GetMyAccountViewRequest(),
        ).account_view
        return Account(
            id=uuid.UUID(account_view.id.value),
            name=account_view.name,
        )
