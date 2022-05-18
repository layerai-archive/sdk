from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class PyTorchModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of PyTorch Models."""

    MODULE_KEYWORD = "torch"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_PYTORCH")

    def save_model_to_directory(
        self, model_object: TrainedModelObject, directory: Path
    ) -> None:
        import mlflow.pytorch

        models_module_path = None

        # Check if the model is a YOLOV5 model
        if model_object.__class__.__name__ == "AutoShape":
            try:
                try:
                    # This is `models` module created cloning the official YoloV5 repo
                    import models  # type: ignore
                    import utils  # type: ignore
                except ModuleNotFoundError:
                    # Fallback: This is `models` module created installing the yolov5 from pypi
                    from yolov5 import models, utils  # type: ignore

                # YOLO models has a wrapper around the pytorch model.
                # We pack the required wrapper modules with the model.
                models_module_path = [
                    Path(models.__file__).parent,
                    Path(utils.__file__).parent,
                ]
            except ModuleNotFoundError:
                raise Exception("Can't save YOLO model. `models` module not found!")

        mlflow.pytorch.save_model(
            model_object, path=directory.as_posix(), code_paths=models_module_path
        )

    def load_model_from_directory(self, directory: Path) -> TrainedModelObject:
        import mlflow.pytorch
        import torch

        return mlflow.pytorch.load_model(
            directory.as_uri(),
            map_location=torch.device("cpu"),
        )
