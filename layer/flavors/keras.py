from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelArtifact

from .base import ModelFlavor


class KerasModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Keras Models."""

    MODULE_KEYWORD = "keras"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_KERAS")

    TOKENIZER_FILE = "tokenizer.pickle"

    def can_interpret_object(self, model_object: ModelArtifact) -> bool:
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
        self, model_object: ModelArtifact, directory: Path
    ) -> None:
        import cloudpickle  # type: ignore
        import keras
        import mlflow.keras

        if isinstance(model_object, keras.preprocessing.text.Tokenizer):
            directory.mkdir(parents=True, exist_ok=True)
            with open(directory / KerasModelFlavor.TOKENIZER_FILE, "wb") as handle:
                cloudpickle.dump(model_object, handle)
        else:
            mlflow.keras.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> ModelArtifact:
        import mlflow.keras

        tokenizer_file = directory / KerasModelFlavor.TOKENIZER_FILE

        if tokenizer_file.exists():
            import cloudpickle

            with open(directory / KerasModelFlavor.TOKENIZER_FILE, "rb") as handle:
                return cloudpickle.load(handle)
        else:
            return mlflow.keras.load_model(directory.as_uri())
