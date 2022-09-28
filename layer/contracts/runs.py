from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class Run:
    """
    Run is created via `layer.init`.

    Data can be logged and labels can attached to a given run.
    """

    id: UUID
    index: int
