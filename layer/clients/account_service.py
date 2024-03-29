import datetime
import uuid
from typing import Optional

import layerapi.api.value.date_pb2
from layerapi.api.entity.account_pb2 import Account as PbAccount
from layerapi.api.entity.account_view_pb2 import AccountView
from layerapi.api.ids_pb2 import AccountId
from layerapi.api.service.account.account_api_pb2 import (
    CreateOrganizationAccountRequest,
    DeleteAccountRequest,
    GetAccountByIdRequest,
    GetAccountByIdResponse,
    GetAccountViewByIdRequest,
    GetAccountViewByNameRequest,
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
        account_view: AccountView = (
            self._account_api.GetAccountViewById(
                GetAccountViewByIdRequest(id=AccountId(value=str(account_id))),
            )
        ).account_view
        return account_view.name

    def get_account_by_name(self, account_name: str) -> Account:
        account_view: AccountView = (
            self._account_api.GetAccountViewByName(
                GetAccountViewByNameRequest(name=account_name),
            )
        ).account_view
        return self._account_from_view(account_view)

    def get_my_account(self) -> Account:
        account_view: AccountView = self._account_api.GetMyAccountView(
            GetMyAccountViewRequest(),
        ).account_view
        return self._account_from_view(account_view)

    def create_organization_account(
        self,
        name: str,
        display_name: Optional[str] = None,
        deletion_allowed: bool = False,
    ) -> Account:
        account_view: AccountView = self._account_api.CreateOrganizationAccount(
            CreateOrganizationAccountRequest(
                name=name,
                display_name=display_name if display_name else name,
                deletion_allowed=deletion_allowed,
            )
        ).account_view
        return self._account_from_view(account_view)

    @staticmethod
    def _account_from_view(account_view: AccountView) -> Account:
        return Account(
            id=uuid.UUID(account_view.id.value),
            name=account_view.name,
        )

    def delete_account(self, account_id: uuid.UUID) -> None:
        self._account_api.DeleteAccount(
            DeleteAccountRequest(account_id=AccountId(value=str(account_id)))
        )

    def get_account_creation_date(self, account_id: uuid.UUID) -> datetime.datetime:
        resp: GetAccountByIdResponse = self._account_api.GetAccountById(
            GetAccountByIdRequest(account_id=AccountId(value=str(account_id)))
        )
        acc: PbAccount = resp.account
        created_date: layerapi.api.value.date_pb2.Date = acc.created_date
        date = datetime.datetime(
            created_date.year_month.year,
            created_date.year_month.month,
            created_date.day,
            tzinfo=datetime.timezone.utc,
        )
        return date
