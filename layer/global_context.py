from dataclasses import dataclass
from typing import List, Optional

from layer import Context
from layer.fabric import Fabric


@dataclass
class GlobalContext:
    project_name: Optional[str]
    fabric: Optional[Fabric]
    active_context: Optional[Context]
    pip_requirements_file: Optional[str]
    pip_packages: Optional[List[str]]


# We store project name, fabric, active context and requirements
# at the process level for use in subsequent calls in the same Python process.
_global_context = GlobalContext(
    project_name=None,
    fabric=None,
    active_context=None,
    pip_requirements_file=None,
    pip_packages=None,
)


def reset_to(project_name: str) -> None:
    if current_project_name() != project_name:
        global _global_context
        _global_context = GlobalContext(
            project_name=project_name,
            fabric=None,
            active_context=None,
            pip_requirements_file=None,
            pip_packages=None,
        )


def set_current_project_name(name: str) -> None:
    global _global_context
    _global_context.project_name = name


def current_project_name() -> Optional[str]:
    return _global_context.project_name


def set_active_context(context: Context) -> None:
    global _global_context
    _global_context.active_context = context


def get_active_context() -> Optional[Context]:
    """
    Returns the active context object set from the active computation. Used in local mode to identify which
    context to log resources to.

    @return:  active context object
    """
    return _global_context.active_context


def set_default_fabric(fabric: Fabric) -> None:
    global _global_context
    _global_context.fabric = fabric


def default_fabric() -> Optional[Fabric]:
    return _global_context.fabric


def set_pip_requirements_file(pip_requirements_file: str) -> None:
    global _global_context
    _global_context.pip_requirements_file = pip_requirements_file


def get_pip_requirements_file() -> Optional[str]:
    return _global_context.pip_requirements_file


def set_pip_packages(packages: List[str]) -> None:
    global _global_context
    _global_context.pip_packages = packages


def get_pip_packages() -> Optional[List[str]]:
    return _global_context.pip_packages
