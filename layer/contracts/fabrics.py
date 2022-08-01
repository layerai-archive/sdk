import enum


@enum.unique
class Fabric(enum.Enum):
    F_LOCAL = "f-local", 0, 0, 0
    F_XXSMALL = "f-xxsmall", 0.5, 2, 0
    F_XSMALL = "f-xsmall", 0.5, 4, 0
    F_SMALL = "f-small", 2, 4, 0
    F_MEDIUM = "f-medium", 3, 14, 0
    F_GPU_SMALL = "f-gpu-small", 3, 48, 1
    F_GPU_LARGE = "f-gpu-large", 3, 48, 2

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, value, cpu, memory, gpu):
        self._value_ = value
        self._cpu = cpu
        self._memory = memory
        self._gpu = gpu

    @property
    def cpu(self) -> float:
        return self._cpu

    @property
    def gpu(self) -> float:
        return self._gpu

    @property
    def memory(self) -> float:
        return self._memory

    @property
    def memory_in_bytes(self) -> float:
        return self._memory * 1000 * 1024 * 1024

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        try:
            cls.__new__(cls, key)
            return True
        except ValueError:
            return False

    @classmethod
    def default(cls) -> "Fabric":
        return Fabric.F_SMALL

    def is_gpu(self) -> bool:
        return "gpu" in self.value

    def __str__(self):
        return f"<{type(self).__name__}.{self.name}: (cpu: {self.cpu!r}, memory: {self.memory!r}, gpu: {self.gpu!r})>"
