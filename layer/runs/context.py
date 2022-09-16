import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.exceptions.exceptions import InitializationException


@dataclass
class RunContext:
    project_full_name: ProjectFullName
    project_id: uuid.UUID
    run_id: uuid.UUID
    labels: Set[str]

    # execution defaults
    fabric: Optional[Fabric] = None
    pip_requirements_file: Optional[str] = None
    pip_packages: Optional[List[str]] = None

    # status
    error: Optional[Exception] = None


# We store full project name, fabric and requirements
# at the process level for use in subsequent calls in the same Python process.
_RUN_CONTEXT: Optional[RunContext] = None


def reset(run_context: Optional[RunContext] = None) -> None:
    global _RUN_CONTEXT
    _RUN_CONTEXT = run_context


def reset_to(
    project_full_name: ProjectFullName,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    labels: Set[str],
) -> None:
    reset(
        RunContext(
            project_full_name=project_full_name,
            project_id=project_id,
            run_id=run_id,
            labels=labels,
        )
    )


def get_run_context() -> RunContext:
    if _RUN_CONTEXT is None:
        raise InitializationException(
            "Please specify the project name with layer.init('account-name/project-name')"
        )
    return _RUN_CONTEXT


def is_initialized() -> bool:
    return get_run_context().project_full_name is not None


def get_project_full_name() -> ProjectFullName:
    return get_run_context().project_full_name


def get_account_name() -> str:
    return get_run_context().project_full_name.account_name


def get_run_id() -> uuid.UUID:
    return get_run_context().run_id


def get_labels() -> Set[str]:
    return get_run_context().labels


def set_default_fabric(fabric: Fabric) -> None:
    get_run_context().fabric = fabric


def default_fabric() -> Optional[Fabric]:
    return get_run_context().fabric


def set_pip_requirements_file(pip_requirements_file: str) -> None:
    get_run_context().pip_requirements_file = pip_requirements_file


def get_pip_requirements_file() -> Optional[str]:
    return get_run_context().pip_requirements_file


def set_pip_packages(packages: Sequence[str]) -> None:
    get_run_context().pip_packages = list(packages)


def get_pip_packages() -> Optional[List[str]]:
    return get_run_context().pip_packages


def set_error(error: Exception) -> None:
    get_run_context().error = error


def get_error() -> Optional[Exception]:
    return get_run_context().error
