import sys
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple

from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.ids_pb2 import RunId

from layer.decorators.definitions import FunctionDefinition

from .project_full_name import ProjectFullName


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro


GetRunsFunction = Callable[[], List[PBRun]]


@dataclass(frozen=True)
class Run:
    """
    Provides access to project runs stored in Layer.

    You can retrieve an instance of this object with :code:`layer.run()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Runs the current project with the given functions
        layer.run([build_dataset, train_model])

    """

    project_full_name: ProjectFullName
    definitions: Sequence[FunctionDefinition] = field(default_factory=list, repr=False)
    files_hash: str = ""
    readme: str = field(repr=False, default="")
    run_id: Optional[RunId] = field(repr=False, default=None)

    @property
    def project_name(self) -> str:
        return self.project_full_name.project_name

    @property
    def account_name(self) -> str:
        return self.project_full_name.account_name

    def with_run_id(self, run_id: RunId) -> "Run":
        return replace(self, run_id=run_id)

    def with_definitions(self, definitions: Sequence[FunctionDefinition]) -> "Run":
        return replace(self, definitions=definitions)
