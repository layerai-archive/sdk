import time
import uuid
from typing import Any, Optional
from uuid import UUID

from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.value.model_flavor_pb2 import ModelFlavor

from layer.clients.layer import LayerClient
from layer.contracts.models import Model
from layer.exceptions.exceptions import UnexpectedModelTypeException
from layer.flavors.utils import get_flavor_for_model
from layer.tracker.ui_progress_tracker import UIRunProgressTracker
from layer.types import ModelObject

from ..contracts.project_full_name import ProjectFullName
from .base_train import BaseTrain


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
        project_full_name: ProjectFullName,
        version: Optional[str],
        train_id: Optional[UUID] = None,
        train_index: Optional[str] = None,
    ):
        self.__layer_client: LayerClient = layer_client
        self.__name: str = name
        self.__project_full_name: ProjectFullName = project_full_name
        self.__version: Optional[str] = str(version)
        self.__train_index: Optional[str] = str(train_index)
        self.__train_id: Optional[ModelTrainId] = (
            ModelTrainId(value=str(train_id)) if train_id is not None else None
        )
        self.__start_train_ts: int  # For computing relative to start metric timestamps
        # Populated at the save of a model train
        self.__flavor: Optional[ModelFlavor.V] = None

    def get_id(self) -> UUID:
        assert self.__train_id
        return UUID(self.__train_id.value)

    def get_version(self) -> str:
        assert self.__version
        return self.__version

    def get_train_index(self) -> str:
        assert self.__train_index
        return self.__train_index

    def save_model(
        self,
        model_object: ModelObject,
        tracker: UIRunProgressTracker,
    ) -> Any:
        assert self.__train_id

        flavor = get_flavor_for_model(model_object)
        if flavor is None:
            raise UnexpectedModelTypeException(type(model_object))
        storage_config = (
            self.__layer_client.model_catalog.get_model_train_storage_configuration(
                self.__train_id
            )
        )
        self.__flavor = flavor.PROTO_FLAVOR
        model = Model(
            self.__name,
            uuid.UUID(self.__train_id.value),
            flavor=flavor,
            storage_config=storage_config,
        )
        self.__layer_client.model_catalog.save_model_object(
            model, model_object, tracker=tracker
        )

    def __start_train(self) -> None:
        if self.__train_id is None:
            self.__train_id = self.__layer_client.model_catalog.create_model_train(
                name=self.__name,
                version=self.__version,
                project_full_name=self.__project_full_name,
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
