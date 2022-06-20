import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

from layer.types import ModelObject


if TYPE_CHECKING:
    from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PBModelFlavor


@dataclass(frozen=True)
class ModelRuntimeObjects:
    model_object: ModelObject
    prediction_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


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
    ) -> "PBModelFlavor.ValueType":
        """Defines the proto flavor that this Model Flavor uses.

        Returns:
            The proto flavor
        """

    def can_interpret_object(self, model_object: ModelObject) -> bool:
        """Checks whether supplied model object has flavor of this class.

        Args:
            model_object: A machine learning model which could be originated from any framework.

        Returns:
            bool: true if this ModelFlavor can interpret the given model instance.
        """
        for hierarchy_class in inspect.getmro(type(model_object)):
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
        model_object: ModelObject,
        directory: Path,
    ) -> None:
        """Defines the method that this Model Flavor uses to save a model to a directory."""

    @abstractmethod
    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        """Defines the method that this Model Flavor uses to load a model from a directory.

        Returns:
             Model object and prediction function.
        """
