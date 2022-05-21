import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.ids_pb2 import ModelTrainId

from layer.clients.layer import LayerClient
from layer.flavors.model_definition import ModelDefinition
from layer.tracker.project_progress_tracker import ProjectProgressTracker

from .base_train import BaseTrain


if TYPE_CHECKING:
    from layer.contracts.models import TrainedModelObject


class Train(BaseTrain):
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

    def __init__(
        self,
        layer_client: LayerClient,
        name: str,
        project_name: str,
        version: Optional[str],
        train_id: Optional[UUID] = None,
        train_index: Optional[str] = None,
    ):
        self.__layer_client: LayerClient = layer_client
        self.__name: str = name
        self.__project_name: str = project_name
        self.__version: Optional[str] = str(version)
        self.__train_index: Optional[str] = str(train_index)
        self.__train_id: Optional[ModelTrainId] = (
            ModelTrainId(value=str(train_id)) if train_id is not None else None
        )
        self.__start_train_ts: int  # For computing relative to start metric timestamps
        # Populated at the save of a model train
        self.__flavor: Optional[ModelVersion.ModelFlavor.V] = None

    def get_id(self) -> UUID:
        assert self.__train_id
        return UUID(self.__train_id.value)

    def get_version(self) -> str:
        assert self.__version
        return self.__version

    def get_train_index(self) -> str:
        assert self.__train_index
        return self.__train_index

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
        assert self.__train_id
        self.__layer_client.model_catalog.log_parameter(
            train_id=self.__train_id, name=str(name), value=str(value)
        )

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Logs a batch of parameters under the current train during model training.

        :param parameters: A dictionary containing all the parameter keys and values. Values are stored as strings by calling str() on each one.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                parameters = { "test_size" : 0.2, "iteration_count": 3}
                train.log_parameters(parameters)

        """
        assert self.__train_id
        parameters = {str(x): str(y) for x, y in parameters.items()}
        self.__layer_client.model_catalog.log_parameters(
            train_id=self.__train_id, parameters=parameters
        )

    def get_parameter(self, name: str) -> Optional[Any]:
        """
        Retrieves a hyperparameter or a training parameter to use during model training.

        :param name: Name of the parameter
        :return: Layer attempts to parse the stored stringified parameter value as a number first, otherwise returned as a string.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                estimators = train.get_parameter("n_estimators")
                max_depth = train.get_parameter("max_depth")
                max_samples = train.get_parameter("max_samples")

        """
        assert self.__train_id
        return self.get_parameters().get(name, None)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves all the hyperparameters and training parameters to use during model training.

        :return: A dictionary containing all the parameter keys and values.

        .. code-block:: python

            def train_model(train: Train, tf: Featureset("transaction_features")):
                # Return previously logged parameters.
                parameters = train.get_parameters()

        """
        assert self.__train_id
        params = self.__layer_client.model_catalog.get_model_train_parameters(
            train_id=self.__train_id
        )
        return {
            param_name: self.__parse_parameter(param_val)
            for param_name, param_val in params.items()
        }

    def save_model(
        self,
        trained_model_obj: "TrainedModelObject",
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> Any:
        if not tracker:
            tracker = ProjectProgressTracker()
        assert self.__train_id
        self.__flavor = self.__infer_flavor(trained_model_obj)
        train_storage_config = (
            self.__layer_client.model_catalog.get_model_train_storage_configuration(
                self.__train_id
            )
        )
        model_definition = ModelDefinition(
            name=self.__name,
            train_id=self.__train_id,
            PROTO_FLAVOR=self.__flavor,
            s3_path=train_storage_config.s3_path,
            credentials=train_storage_config.credentials,
        )
        self.__layer_client.model_catalog.save_model(
            model_definition, trained_model_obj=trained_model_obj, tracker=tracker
        )

    def __parse_parameter(self, value: str) -> Any:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def __start_train(self) -> None:
        if self.__train_id is None:
            self.__train_id = self.__layer_client.model_catalog.create_model_train(
                name=self.__name,
                version=self.__version,
                project_name=self.__project_name,
            )
        self.__layer_client.model_catalog.start_model_train(
            train_id=self.__train_id,
        )
        self.__start_train_ts = int(time.time())

    def __complete_train(self) -> None:
        assert self.__train_id
        self.__layer_client.model_catalog.complete_model_train(
            self.__train_id, self.__flavor
        )

    def __infer_flavor(
        self, model_obj: "TrainedModelObject"
    ) -> ModelVersion.ModelFlavor:
        return self.__layer_client.model_catalog.infer_flavor(model_obj)

    def __enter__(self) -> Any:
        self.__start_train()
        return self

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        # From the docs we don't need to re-raise the exception as it will be thrown after
        # the execution of this method. We'd only proceed if there are no exceptions.
        # In happy path, all three parameters would be None.
        # https://docs.python.org/3/reference/datamodel.html#object.__exit__
        if exception_type is None:
            self.__complete_train()
