from types import TracebackType
from typing import Optional

from layer.dataset_build import DatasetBuild
from layer.train import Train


class Context:
    """
    Provides access to variables within the pipeline execution.

    An instance of this class is automatically passed in as an argument to the user-defined functions.
    It must always be listed as the first argument in the user-defined functions.

    This class should not be initialized by end-users.
    """

    def __init__(
        self,
        train: Optional[Train] = None,
        dataset_build: Optional[DatasetBuild] = None,
    ) -> None:
        self._train: Optional[Train] = train
        self._dataset_build: Optional[DatasetBuild] = dataset_build

    def train(self) -> Optional[Train]:
        """
        Retrieves the active Layer train object.

        :return: Represents the current train of the model, passed by Layer when the training of the model starts.

        .. code-block:: python

            # Get train object from Layer Context.
            train = context.train()
        """
        return self._train

    def dataset_build(self) -> Optional[DatasetBuild]:
        """
        Retrieves the active Layer dataset build object.

        :return: Represents the current dataset build of the dataset, passed by Layer when building of the dataset starts.
        """
        return self._dataset_build

    def with_train(self, train: Optional[Train]) -> None:
        self._train = train

    def with_dataset_build(self, dataset_build: Optional[DatasetBuild]) -> None:
        self._dataset_build = dataset_build

    def close(self) -> None:
        pass

    def __enter__(self) -> "Context":
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
