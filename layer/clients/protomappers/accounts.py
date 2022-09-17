from layerapi import api

from layer.contracts import ids
from layer.contracts.accounts import Account


def to_account_id(id: ids.AccountId) -> api.AccountId:
    return api.AccountId(value=str(id))


def from_account_id(id: api.AccountId) -> ids.AccountId:
    return ids.AccountId(id.value)


def from_account_view(val: api.AccountView) -> Account:
    return Account(
        id=from_account_id(val.id),
        name=val.name,
    )
