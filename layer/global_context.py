from dataclasses import dataclass
from typing import List, Optional

from .context import Context
from .contracts.fabrics import Fabric


@dataclass
class GlobalContext:
    project_name: Optional[str]
    fabric: Optional[Fabric]
    active_context: Optional[Context]
    pip_requirements_file: Optional[str]
    pip_packages: Optional[List[str]]


# We store project name, fabric, active context and requirements
# at the process level for use in subsequent calls in the same Python process.
_GLOBAL_CONTEXT = GlobalContext(
    project_name=None,
    fabric=None,
    active_context=None,
    pip_requirements_file=None,
    pip_packages=None,
)


def reset_to(project_name: Optional[str]) -> None:
    if current_project_name() != project_name:
        global _GLOBAL_CONTEXT
        _GLOBAL_CONTEXT = GlobalContext(
            project_name=project_name,
            fabric=None,
            active_context=None,
            pip_requirements_file=None,
            pip_packages=None,
        )


def set_current_project_name(name: Optional[str]) -> None:
    _GLOBAL_CONTEXT.project_name = name


def current_project_name() -> Optional[str]:
    return _GLOBAL_CONTEXT.project_name


def set_active_context(context: Context) -> None:
    _GLOBAL_CONTEXT.active_context = context


def reset_active_context() -> None:
    _GLOBAL_CONTEXT.active_context = None


def get_active_context() -> Optional[Context]:
    """
    Returns the active context object set from the active computation. Used in local mode to identify which
    context to log resources to.

    @return:  active context object
    """
    return _GLOBAL_CONTEXT.active_context


def set_default_fabric(fabric: Fabric) -> None:
    _GLOBAL_CONTEXT.fabric = fabric


def default_fabric() -> Optional[Fabric]:
    return _GLOBAL_CONTEXT.fabric


def set_pip_requirements_file(pip_requirements_file: str) -> None:
    _GLOBAL_CONTEXT.pip_requirements_file = pip_requirements_file


def get_pip_requirements_file() -> Optional[str]:
    return _GLOBAL_CONTEXT.pip_requirements_file


def set_pip_packages(packages: List[str]) -> None:
    _GLOBAL_CONTEXT.pip_packages = packages


def get_pip_packages() -> Optional[List[str]]:
    return _GLOBAL_CONTEXT.pip_packages
