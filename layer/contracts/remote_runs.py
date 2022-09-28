import enum
import uuid
from dataclasses import dataclass


@enum.unique
class TaskType(enum.IntEnum):
    MODEL_TRAIN = 1
    DATASET_BUILD = 2


@enum.unique
class RunStatus(enum.IntEnum):
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3


@dataclass(frozen=True)
class RemoteRun:
    """
    Provides access to project remote runs stored in Layer.

    You can retrieve an instance of this object with :code:`layer.run()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Runs the current project with the given functions
        layer.run([build_dataset, train_model])

    """

    id: uuid.UUID
