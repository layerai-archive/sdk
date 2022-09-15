import uuid
from dataclasses import dataclass
from typing import List, Optional, Set

from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName


@dataclass
class RunContext:
    project_full_name: Optional[ProjectFullName]
    project_id: Optional[uuid.UUID]
    run_id: Optional[
        uuid.UUID
    ]  # generated from RUN_ID_ENV_VARIABLE, run_index or just new
    label_names: Optional[Set[str]]

    # execution config
    fabric: Optional[Fabric]
    pip_requirements_file: Optional[str]
    pip_packages: Optional[List[str]]

    # ui state
    # We show a message to the user if their installed layer version is outdated.
    # We want to avoid showing this in case we already checked the version or we have already shown this message.
    has_shown_update_message: bool
    # Similar to above, but for supported Python version.
    has_shown_python_version_message: bool


def _new_run_context() -> RunContext:
    return RunContext(
        project_full_name=None,
        project_id=None,
        run_id=None,
        label_names=None,
        fabric=None,
        pip_requirements_file=None,
        pip_packages=None,
        has_shown_update_message=False,
        has_shown_python_version_message=False,
    )


# We store full project name, fabric and requirements
# at the process level for use in subsequent calls in the same Python process.
_RUN_CONTEXT = _new_run_context()


def reset_to(
    project_full_name: Optional[ProjectFullName],
    label_names: Optional[Set[str]] = None,
) -> None:
    global _RUN_CONTEXT
    _RUN_CONTEXT = _new_run_context()
    _RUN_CONTEXT.project_full_name = project_full_name
    _RUN_CONTEXT.label_names = label_names


def set_current_project_full_name(name: Optional[ProjectFullName]) -> None:
    _RUN_CONTEXT.project_full_name = name


def current_project_full_name() -> Optional[ProjectFullName]:
    return _RUN_CONTEXT.project_full_name


def current_account_name() -> Optional[str]:
    return (
        _RUN_CONTEXT.project_full_name.account_name
        if _RUN_CONTEXT.project_full_name
        else None
    )


def current_label_names() -> Set[str]:
    return _RUN_CONTEXT.label_names if _RUN_CONTEXT.label_names else set()


def set_default_fabric(fabric: Fabric) -> None:
    _RUN_CONTEXT.fabric = fabric


def default_fabric() -> Optional[Fabric]:
    return _RUN_CONTEXT.fabric


def set_pip_requirements_file(pip_requirements_file: str) -> None:
    _RUN_CONTEXT.pip_requirements_file = pip_requirements_file


def get_pip_requirements_file() -> Optional[str]:
    return _RUN_CONTEXT.pip_requirements_file


def set_pip_packages(packages: List[str]) -> None:
    _RUN_CONTEXT.pip_packages = packages


def get_pip_packages() -> Optional[List[str]]:
    return _RUN_CONTEXT.pip_packages


def set_has_shown_update_message(shown: bool) -> None:
    _RUN_CONTEXT.has_shown_update_message = shown


def has_shown_update_message() -> bool:
    return _RUN_CONTEXT.has_shown_update_message


def set_has_shown_python_version_message(shown: bool) -> None:
    _RUN_CONTEXT.has_shown_python_version_message = shown


def has_shown_python_version_message() -> bool:
    return _RUN_CONTEXT.has_shown_python_version_message
