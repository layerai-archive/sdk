from pathlib import Path

import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class ScikitLearnModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Scikit Learn Models."""

    MODULE_KEYWORD = "sklearn"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_SKLEARN

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
    ) -> None:
        import mlflow.sklearn

        # Added serialization_format specifically to disable pickle5 which causes errors on Python 3.7
        return mlflow.sklearn.save_model(
            model_object, path=directory.as_posix(), serialization_format="pickle"
        )

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(directory.as_uri())
        return ModelRuntimeObjects(
            model, lambda input_df: self.__predict(model, input_df)
        )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        prediction_np_array = model.predict(input_df)  # type: ignore
        return pd.DataFrame(prediction_np_array)
