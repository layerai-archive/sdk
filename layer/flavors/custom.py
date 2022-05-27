from abc import abstractmethod
from pathlib import Path
from typing import Any

import cloudpickle  # type: ignore
import pandas
from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelArtifact

from .base import ModelFlavor


class CustomModel:
    """
    A generic model that evaluates inputs and produces outputs.
    """

    def __init__(self) -> None:
        """
        Initializes the custom  model
        """

    @abstractmethod
    def predict(self, model_input: pandas.DataFrame) -> pandas.DataFrame:
        """
        Evaluates an input for this model and produces an output.

        :param model_input: A pandas.DataFrame as input for the model to evaluate.
        :return: Model output as a pandas.DataFrame
        """


class CustomModelFlavor(ModelFlavor):
    """A model flavor implementation which enables custom model implementation for Layer users."""

    MODULE_KEYWORD = "layer"
    MODEL_PICKLE_FILE = "custom_model.pkl"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.MODEL_FLAVOR_CUSTOM

    def save_model_to_directory(
        self,
        model_object: Any,
        directory: Path,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / self.MODEL_PICKLE_FILE, mode="wb") as file:
            cloudpickle.dump(model_object, file)

    def load_model_from_directory(self, directory: Path) -> ModelArtifact:
        with open(directory / self.MODEL_PICKLE_FILE, mode="rb") as file:
            custom_model = cloudpickle.load(file)
            return custom_model
