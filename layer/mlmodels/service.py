import importlib
from logging import Logger
from typing import TYPE_CHECKING, Optional, Tuple

from yarl import URL

from layer.exceptions.exceptions import (
    LayerClientException,
    UnexpectedModelTypeException,
)
from layer.projects.tracker.project_progress_tracker import ProjectProgressTracker
from layer.projects.tracker.resource_transfer_state import ResourceTransferState

from ..api.entity.model_version_pb2 import ModelVersion  # pylint: disable=unused-import
from .flavors import (
    PROTO_TO_PYTHON_OBJECT_FLAVORS,
    PYTHON_CLASS_NAME_TO_PROTO_FLAVORS,
    PYTHON_FLAVORS,
    ModelFlavor,
)
from .flavors.model_definition import ModelDefinition


if TYPE_CHECKING:
    from . import ModelObject


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
        model_object: "ModelObject",
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
            flavor.save(
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
    ) -> "ModelObject":
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
            model_definition.proto_flavor
        )

        self.logger.debug(f"Loading model {model_definition.model_name}")
        module = importlib.import_module(flavor.metadata.module_name)
        model_flavor_class = getattr(module, flavor.metadata.class_name)
        return model_flavor_class(no_cache=no_cache).load(
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
        model_object: "ModelObject",
        logger: Logger,
    ) -> Tuple["ModelVersion.ModelFlavor.V", ModelFlavor]:
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
        flavor = MLModelService.__check_and_get_flavor(model_object, logger)
        if flavor is None:
            raise UnexpectedModelTypeException(type(model_object))
        return flavor

    @staticmethod
    def get_model_flavor_from_proto(
        proto_flavor: "ModelVersion.ModelFlavor.V",
    ) -> ModelFlavor:
        if proto_flavor not in PROTO_TO_PYTHON_OBJECT_FLAVORS:
            raise LayerClientException(f"Unexpected model flavor {type(proto_flavor)}")
        return PROTO_TO_PYTHON_OBJECT_FLAVORS[proto_flavor]

    @staticmethod
    def __check_and_get_flavor(
        model_object: "ModelObject",
        logger: Logger,
    ) -> Optional[Tuple["ModelVersion.ModelFlavor.V", ModelFlavor]]:
        matching_flavor: Optional[ModelFlavor] = None
        for flavor in PYTHON_FLAVORS:
            if flavor.can_interpret_object(model_object):
                matching_flavor = flavor
                break
        logger.info(f"Matching flavor: {matching_flavor}")
        if matching_flavor is None:
            return None
        flavor_name = type(matching_flavor).__name__
        proto_flavor = PYTHON_CLASS_NAME_TO_PROTO_FLAVORS[flavor_name]
        logger.info(f"flavor name: {flavor_name}, proto flavor: {proto_flavor}")
        return proto_flavor, matching_flavor
