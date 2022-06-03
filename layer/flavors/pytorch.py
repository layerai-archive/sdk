from pathlib import Path

import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class PyTorchModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of PyTorch Models."""

    MODULE_KEYWORD = "torch"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_PYTORCH

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
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

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.pytorch
        import torch

        model = mlflow.pytorch.load_model(
            directory.as_uri(),
            map_location=torch.device("cpu"),
        )
        return ModelRuntimeObjects(
            model, lambda input_df: self.__predict(model, input_df)
        )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        from mlflow.pytorch import _PyTorchWrapper

        model = _PyTorchWrapper(model)
        return model.predict(input_df)  # type: ignore
