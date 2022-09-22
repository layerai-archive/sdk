import enum


@enum.unique
class Fabric(enum.Enum):
    F_LOCAL = ("f-local", 0, 0, 0)
    F_XXSMALL = ("f-xxsmall", 0.5, 2, 0)
    F_XSMALL = ("f-xsmall", 0.5, 4, 0)
    F_SMALL = ("f-small", 2, 4, 0)
    F_MEDIUM = ("f-medium", 3, 14, 0)
    F_GPU_SMALL = ("f-gpu-small", 3, 48, 1)
    F_GPU_LARGE = ("f-gpu-large", 3, 48, 2)

    def __new__(cls, value: str, cpu: float, memory: float, gpu: float) -> "Fabric":
        entry = object.__new__(cls)
        entry._value_ = value
        entry._cpu = cpu  # type:ignore
        entry._memory = memory  # type:ignore
        entry._gpu = gpu  # type:ignore
        return entry

    @classmethod
    def find(cls, value: str) -> "Fabric":
        for fabric in cls.__members__.values():
            if fabric.value == value:
                return fabric
        raise ValueError(f"'{cls.__name__}' enum not found for '{value}'")

    @property
    def cpu(self) -> float:
        return self._cpu  # type:ignore

    @property
    def gpu(self) -> float:
        return self._gpu  # type:ignore

    @property
    def memory(self) -> float:
        return self._memory  # type:ignore

    @property
    def memory_in_bytes(self) -> float:
        return self._memory * 1000 * 1024 * 1024  # type:ignore

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        try:
            cls.__new__(cls, key)  # type:ignore # pylint: disable=E1120
            return True
        except ValueError:
            return False

    @classmethod
    def default(cls) -> "Fabric":
        return Fabric.F_SMALL

    def is_gpu(self) -> bool:
        return "gpu" in self.value

    def __str__(self) -> str:
        return f"<{type(self).__name__}.{self.name}: (cpu: {self.cpu!r}, memory: {self.memory!r}, gpu: {self.gpu!r})>"
