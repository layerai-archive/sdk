from pathlib import Path

import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


LAYER_CACHED_TF_MODELS = {}


class TensorFlowModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of TensorFlow Models."""

    MODULE_KEYWORD = "tensorflow"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_TENSORFLOW

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
    ) -> None:
        import mlflow.tensorflow
        import tensorflow  # type: ignore

        tmp_save_path = directory / "tensorflowmodel"
        tensorflow.saved_model.save(model_object, tmp_save_path)

        mlflow.tensorflow.save_model(
            tf_saved_model_dir=tmp_save_path,
            tf_meta_graph_tags=None,
            tf_signature_def_key="serving_default",
            path=directory.as_posix(),
        )

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.tensorflow

        model = mlflow.tensorflow.load_model(directory.as_uri())
        return ModelRuntimeObjects(
            model, lambda input_df: self.__predict(directory, input_df)
        )

    @staticmethod
    def __predict(directory: Path, input_df: pd.DataFrame) -> pd.DataFrame:
        cache_key = str(directory)
        if cache_key not in LAYER_CACHED_TF_MODELS:
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model(directory.as_uri())
            LAYER_CACHED_TF_MODELS[cache_key] = model
        model = LAYER_CACHED_TF_MODELS[cache_key]
        return model.predict(input_df)
