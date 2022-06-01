import pickle  # nosec
from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas
from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


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
    MODEL_SOURCE_FILE = "custom_model.py"
    MODEL_CONFIG_FILE = "custom_model.config"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.MODEL_FLAVOR_CUSTOM

    def save_model_to_directory(
        self,
        model_object: Any,
        directory: Path,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        # Store class config
        with open(directory / self.MODEL_CONFIG_FILE, mode="w") as file:
            custom_model_class = type(model_object)
            class_config = (
                custom_model_class.__module__ + "." + custom_model_class.__name__
            )
            file.write(class_config)

        # Store class definition
        import inspect

        custom_class_model = type(model_object)
        class_model_source_code = inspect.getsource(custom_class_model)
        with open(directory / self.MODEL_SOURCE_FILE, mode="w") as file:
            class_model_source_code = (
                "import layer\nfrom layer import CustomModel\n\n"
                + class_model_source_code
            )
            file.write(class_model_source_code)

        # Store model itself
        with open(directory / self.MODEL_PICKLE_FILE, mode="wb") as file:
            pickle.dump(model_object, file, protocol=pickle.DEFAULT_PROTOCOL)

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        # Load model config
        with open(directory / self.MODEL_CONFIG_FILE, mode="r") as file:
            model_config = file.readline()
            model_name = model_config.split(".")[-1]
            model_module_name = ".".join(model_config.split(".")[:-1])

        # Load and register custom class definition
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            model_name, directory / self.MODEL_SOURCE_FILE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[model_module_name] = module

        # Load model itself
        with open(directory / self.MODEL_PICKLE_FILE, mode="rb") as file:
            model = pickle.load(file)  # nosec
            return ModelRuntimeObjects(
                model, lambda input_df: self.__predict(model, input_df)
            )

    @staticmethod
    def __predict(model: ModelObject, input_df: pandas.DataFrame) -> pandas.DataFrame:
        return model.predict(input_df)  # type: ignore
