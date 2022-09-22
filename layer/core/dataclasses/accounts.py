from dataclasses import dataclass

from . import ids


@dataclass(frozen=True)
class Account:
    id: ids.AccountId
    name: str


@dataclass(frozen=True)
class User:
    id: ids.UserId
    account_id: ids.AccountId
    name: str
    email: str
    first_name: str
    last_name: str
