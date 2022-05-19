from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from layer.tracker.project_progress_tracker import RunProgressTracker


if TYPE_CHECKING:
    from layer.types import ModelArtifact


class BaseTrain:
    """
    Allows the user to control the state of a model training.

    The Train object methods can then be called to log parameters or metrics during training.

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

    def log_parameter(self, name: str, value: Any) -> None:
        """
        Logs a parameter under the current train during model training.

        :param name: Name of the parameter.
        :param value: Value to log, stored as a string by calling `str()` on it.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                # Split dataset into training set and test set
                test_size = 0.2
                train.log_parameter("test_size", test_size)

        """

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Logs a batch of parameters under the current train during model training.

        :param parameters: A dictionary containing all the parameter keys and values. Values are stored as strings by calling str() on each one.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                parameters = { "test_size" : 0.2, "iteration_count": 3}
                train.log_parameters(parameters)

        """

    def get_parameter(self, name: str) -> Optional[Any]:
        """
        Retrieves a training parameter to use during model training.

        :param name: Name of the parameter
        :return: Layer attempts to parse the stored stringified parameter value as a number first, otherwise returned as a string.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                estimators = train.get_parameter("n_estimators")
                max_depth = train.get_parameter("max_depth")
                max_samples = train.get_parameter("max_samples")

        """

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves all the training parameters to use during model training.

        :return: A dictionary containing all the parameter keys and values.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                # Return previously logged parameters.
                parameters = train.get_parameters()

        """

    def save_model(
        self,
        trained_model_obj: "ModelArtifact",
        tracker: Optional[RunProgressTracker] = None,
    ) -> Any:
        pass

    def __parse_parameter(self, value: str) -> Any:
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
