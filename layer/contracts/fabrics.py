import enum


@enum.unique
class Fabric(enum.Enum):
    F_LOCAL = "f-local"
    F_XXSMALL = "f-xxsmall"
    F_XSMALL = "f-xsmall"
    F_SMALL = "f-small"
    F_MEDIUM = "f-medium"
    F_GPU_SMALL = "f-gpu-small"

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        try:
            cls.__new__(cls, key)
            return True
        except ValueError:
            return False

    @classmethod
    def default(cls) -> str:
        return Fabric.F_SMALL.value

    def is_gpu(self) -> bool:
        return "gpu" in self.value
