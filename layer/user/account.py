from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class Account:
    id: UUID
    name: str
