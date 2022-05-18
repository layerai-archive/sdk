import abc
import uuid
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID


@dataclass(frozen=True)
class Account:
    id: UUID
    name: str


@dataclass(frozen=True)
class User(abc.ABC):
    name: str
    email: str
    first_name: str
    last_name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    account_id: Optional[uuid.UUID] = field(default_factory=uuid.uuid4)
