from pathlib import Path

import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class KerasModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Keras Models."""

    MODULE_KEYWORD = "keras"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_KERAS

    TOKENIZER_FILE = "tokenizer.pickle"

    def can_interpret_object(self, model_object: ModelObject) -> bool:
        # Check if model is a tensorflow keras
        try:
            import tensorflow.keras.models  # type: ignore

            if isinstance(model_object, tensorflow.keras.models.Model):
                return True
        except ImportError:
            pass

        # Check if model is a pure keras
        try:
            import keras.engine.network  # type: ignore

            if isinstance(model_object, keras.engine.network.Network):
                return True
        except ImportError:
            pass

        # Check if model is a tensorflow/pure keras tokenizer
        try:
            import keras

            if isinstance(model_object, keras.preprocessing.text.Tokenizer):
                return True
        except ImportError:
            pass

        return False

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
    ) -> None:
        import keras
        import mlflow.keras

        from .. import cloudpickle

        if isinstance(model_object, keras.preprocessing.text.Tokenizer):
            directory.mkdir(parents=True, exist_ok=True)
            with open(directory / KerasModelFlavor.TOKENIZER_FILE, "wb") as handle:
                cloudpickle.dump(model_object, handle)  # type: ignore
        else:
            mlflow.keras.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.keras

        tokenizer_file = directory / KerasModelFlavor.TOKENIZER_FILE

        if tokenizer_file.exists():
            from .. import cloudpickle

            with open(directory / KerasModelFlavor.TOKENIZER_FILE, "rb") as handle:
                model = cloudpickle.load(handle)  # type: ignore
                return ModelRuntimeObjects(
                    model, lambda input_df: self.__predict(model, input_df)
                )
        else:
            model = mlflow.keras.load_model(directory.as_uri())
            return ModelRuntimeObjects(
                model, lambda input_df: self.__predict(model, input_df)
            )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        #  https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        predictions = model.predict(input_df)  # type: ignore
        return pd.DataFrame(predictions)
