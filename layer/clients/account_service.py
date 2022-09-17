import datetime
from typing import Optional

from layerapi import api

from layer.config import ClientConfig
from layer.contracts import ids
from layer.contracts.accounts import Account
from layer.utils.grpc.channel import get_grpc_channel

from .protomappers import accounts as account_proto_mapper


class AccountServiceClient:
    _account_api: api.AccountAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "AccountServiceClient":
        client = AccountServiceClient()
        channel = get_grpc_channel(config)
        client._account_api = api.AccountAPIStub(  # pylint: disable=protected-access
            channel
        )
        return client

    async def get_account_by_id(self, account_id: ids.AccountId) -> Account:
        response = await self._account_api.get_account_view_by_id(
            id=account_proto_mapper.to_account_id(account_id),
        )
        return account_proto_mapper.from_account_view(response.account_view)

    async def get_account_by_name(self, account_name: str) -> Account:
        response = await self._account_api.get_account_view_by_name(
            name=account_name,
        )
        return account_proto_mapper.from_account_view(response.account_view)

    async def get_my_account(self) -> Account:
        response = await self._account_api.get_my_account_view()
        return account_proto_mapper.from_account_view(response.account_view)

    async def create_organization_account(
        self,
        name: str,
        display_name: Optional[str] = None,
        deletion_allowed: bool = False,
    ) -> Account:
        response = await self._account_api.create_organization_account(
            name=name,
            display_name=display_name if display_name else name,
            deletion_allowed=deletion_allowed,
        )
        return account_proto_mapper.from_account_view(response.account_view)

    async def delete_account(self, account_id: ids.AccountId) -> None:
        await self._account_api.delete_account(
            account_id=account_proto_mapper.to_account_id(account_id),
        )

    async def get_account_creation_date(
        self, account_id: ids.AccountId
    ) -> datetime.date:
        response = await self._account_api.get_account_by_id(
            account_id=account_proto_mapper.to_account_id(account_id),
        )
        date = response.account.created_date
        return datetime.datetime(
            date.year_month.year,
            date.year_month.month,
            date.day,
            tzinfo=datetime.timezone.utc,
        )
