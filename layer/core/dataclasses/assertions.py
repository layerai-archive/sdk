from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass(frozen=True)
class Assertion:
    name: str
    values: List[Any]
    function: Callable[..., Any]

    def __str__(self) -> str:
        values_str = []
        for value in self.values:
            if callable(value):
                values_str.append(value.__name__)
            else:
                values_str.append(str(value))
        return f"{self.name}({', '.join(values_str)})"
