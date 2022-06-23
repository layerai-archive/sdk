import inspect
from pathlib import Path
from typing import Any, List, Optional


def get_base_module_list(value: Any) -> List[str]:
    return [
        inspect.getmodule(clazz).__name__ for clazz in inspect.getmro(type(value))  # type: ignore
    ]


def has_allowed_extension(file: Path, allowed_extensions: Optional[List[str]]) -> bool:
    if allowed_extensions is None:
        allowed_extensions = []
    extension = file.suffix.lower()
    for allowed_extension in allowed_extensions:
        if extension == allowed_extension:
            return True
    return False
