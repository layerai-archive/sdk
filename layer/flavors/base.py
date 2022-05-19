import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path

from layerapi.api.entity.model_version_pb2 import (  # pylint: disable=unused-import
    ModelVersion,
)

from layer.types import ModelArtifact


class ModelFlavor(metaclass=ABCMeta):
    """Represents a machine learning model flavor for a specific framework.

    Implementations provide methods for checking for membership, saving and loading
    of machine learning models within their flavor.
    """

    @abstractproperty
    def MODULE_KEYWORD(self) -> str:  # pylint: disable=invalid-name
        """Defines a keyword as part of an object's module name that matches this flavor.

        Returns:
            keyword: a str
        """

    @abstractproperty
    def PROTO_FLAVOR(  # pylint: disable=invalid-name
        self,
    ) -> "ModelVersion.ModelFlavor":
        """Defines the proto flavor that this Model Flavor uses.

        Returns:
            The proto flavor
        """

    def can_interpret_object(self, model_artifact: ModelArtifact) -> bool:
        """Checks whether supplied model object has flavor of this class.

        Args:
            model_artifact: A machine learning model which could be originated from any framework.

        Returns:
            bool: true if this ModelFlavor can interpret the given model instance.
        """
        for hierarchy_class in inspect.getmro(type(model_artifact)):
            parent_module = inspect.getmodule(hierarchy_class)
            if (
                parent_module is not None
                and self.MODULE_KEYWORD in parent_module.__name__
            ):
                return True

        return False

    @abstractmethod
    def save_model_to_directory(
        self,
        model_artifact: ModelArtifact,
        directory: Path,
    ) -> None:
        """Defines the method that this Model Flavor uses to save a model to a directory.

        Returns:
             A callable to save the model.
        """

    @abstractmethod
    def load_model_from_directory(self, directory: Path) -> ModelArtifact:
        """Defines the method that this Model Flavor uses to load a model from a directory.

        Returns:
             A callable to load the model.
        """
