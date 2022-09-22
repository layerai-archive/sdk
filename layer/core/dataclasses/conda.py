from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, AnyStr, Dict

import yaml  # type: ignore


@dataclass(frozen=True)
class CondaEnv:
    environment: Dict[str, Any] = field(default_factory=dict)

    def dump_to_file(self, path: Path) -> None:
        with open(path, mode="w", encoding="utf8") as conda:
            yaml.dump(self.environment, conda, default_flow_style=False)

    @classmethod
    def load_from_file(cls, path: Path) -> "CondaEnv":
        with open(path, "r") as file:
            return cls.load_from_stream(file)

    @classmethod
    def load_from_stream(cls, stream: IO[AnyStr]) -> "CondaEnv":
        return cls(environment=yaml.safe_load(stream))
