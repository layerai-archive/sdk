import importlib
from logging import Logger
from typing import TYPE_CHECKING, Optional

from layerapi.api.entity.model_version_pb2 import ModelVersion
from yarl import URL

from layer.contracts.runs import ResourceTransferState
from layer.exceptions.exceptions import (
    LayerClientException,
    UnexpectedModelTypeException,
)
from layer.flavors import ModelFlavor
from layer.flavors.model_definition import ModelDefinition
from layer.flavors.utils import get_flavor_for_model, get_flavor_for_proto
from layer.tracker.project_progress_tracker import ProjectProgressTracker


if TYPE_CHECKING:
    from layer.contracts.models import TrainedModelObject


class MLModelService:
    """
    Handles ML model lifecycle within the application. Users of this service can
    store/load/delete ML models.
    """

    def __init__(self, logger: Logger, s3_endpoint_url: Optional[URL] = None):
        self.logger = logger
        self._s3_endpoint_url = s3_endpoint_url

    # pytype: disable=annotation-type-mismatch # https://github.com/google/pytype/issues/640
    def store(
        self,
        model_definition: ModelDefinition,
        model_object: "TrainedModelObject",
        flavor: ModelFlavor,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        # pytype: enable=annotation-type-mismatch
        """
        Stores given model object along with its definition to the backing storage.
        The metadata written to db and used later on whilst loading the ML model.

        Args:
            model_definition: Model metadata object which describes the model instance
            model_object: Model object to be stored
            flavor: Corresponding flavor information of the model object
        """
        if not tracker:
            tracker = ProjectProgressTracker()
        self.logger.debug(
            f"Saving user model {model_definition.model_name}({model_object})"
        )
        try:
            self.logger.debug(f"Writing model {model_definition}")
            flavor.save_to_s3(
                model_definition,
                model_object,
                s3_endpoint_url=self._s3_endpoint_url,
                tracker=tracker,
            )
            self.logger.debug(
                f"User model {model_definition.model_name} saved successfully"
            )
        except Exception as ex:
            raise LayerClientException(f"Error while storing model, {ex}")

    def retrieve(
        self,
        model_definition: ModelDefinition,
        no_cache: bool = False,
        state: Optional[ResourceTransferState] = None,
    ) -> "TrainedModelObject":
        """
        Retrieves the given model definition from the storage and returns the actual
        model object

        Args:
            model_definition: Model metadata object which describes the model instance
            no_cache: if True, force model fetch from the remote location
        Returns:
            Loaded model object

        """
        self.logger.debug(
            f"User requested to load model {model_definition.model_name} "
        )
        flavor: ModelFlavor = self.get_model_flavor_from_proto(
            model_definition.PROTO_FLAVOR
        )

        self.logger.debug(f"Loading model {model_definition.model_name}")
        module = importlib.import_module(flavor.metadata.module_name)
        model_flavor_class = getattr(module, flavor.metadata.class_name)
        return model_flavor_class(no_cache=no_cache).load_from_s3(
            model_definition=model_definition,
            s3_endpoint_url=self._s3_endpoint_url,
            state=state,
        )

    def delete(self, model_definition: ModelDefinition) -> None:
        """
        Deletes the model along with its metadata from the storage

        Args:
            model_definition: Model metadata object which describes the model instance
        """
        self.logger.debug(
            f"User requested to delete model {model_definition.model_name}"
        )

    @staticmethod
    def get_model_flavor(
        model_object: "TrainedModelObject",
        logger: Logger,
    ) -> ModelFlavor:
        """
        Checks if given model objects has a known model flavor and returns
        the flavor if there is a match.

        Args:
            model_object: User supplied model object

        Returns:
            The corresponding model flavor if there is match

        Raises:
            LayerException if user provided object does not have a known flavor.

        """
        flavor = get_flavor_for_model(model_object)
        if flavor is None:
            raise UnexpectedModelTypeException(type(model_object))
        return flavor

    @staticmethod
    def get_model_flavor_from_proto(
        proto_flavor: ModelVersion.ModelFlavor,
    ) -> ModelFlavor:
        flavor = get_flavor_for_proto(proto_flavor)
        if flavor is None:
            raise LayerClientException(f"Unexpected model flavor {type(proto_flavor)}")
        return flavor
