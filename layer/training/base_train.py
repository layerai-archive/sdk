from typing import TYPE_CHECKING, Any
from uuid import UUID

from layer.tracker.ui_progress_tracker import UIRunProgressTracker


if TYPE_CHECKING:
    from layer.types import ModelObject


class BaseTrain:
    """
    Allows the user to control the state of a model training.

    This class should never be instantiated directly by end users.

    .. code-block:: python

        def train_model(train: Train, tf: Featureset("transaction_features")):
            # We create the training and label data
            train_df = tf.to_pandas()
            X = train_df.drop(["transactionId", "is_fraud"], axis=1)
            Y = train_df["is_fraud"]
            # user model training will happen below [...]
    """

    def get_id(self) -> UUID:
        pass

    def get_version(self) -> str:
        pass

    def get_train_index(self) -> str:
        pass

    def save_model(
        self,
        trained_model_obj: "ModelObject",
        tracker: UIRunProgressTracker,
    ) -> Any:
        pass

    def __start_train(self) -> None:
        pass

    def __complete_train(self) -> None:
        pass

    def __enter__(self) -> Any:
        pass

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        pass
