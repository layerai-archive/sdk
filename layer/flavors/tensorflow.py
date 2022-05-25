from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelArtifact

from .base import ModelFlavor


class TensorFlowModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of TensorFlow Models."""

    MODULE_KEYWORD = "tensorflow"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_TENSORFLOW")

    def save_model_to_directory(
        self, model_object: ModelArtifact, directory: Path
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

    def load_model_from_directory(self, directory: Path) -> ModelArtifact:
        import mlflow.tensorflow

        return mlflow.tensorflow.load_model(directory.as_uri())
