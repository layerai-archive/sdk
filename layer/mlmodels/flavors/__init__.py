from typing import Dict, List

from ...api.entity.model_version_pb2 import ModelVersion
from .flavor import (
    CatBoostModelFlavor,
    HuggingFaceModelFlavor,
    KerasModelFlavor,
    LightGBMModelFlavor,
    ModelFlavor,
    PyTorchModelFlavor,
    ScikitLearnModelFlavor,
    TensorFlowModelFlavor,
    XGBoostModelFlavor,
)


# mypy proto enum typing issue https://github.com/protocolbuffers/protobuf/issues/8175
PROTO_TO_PYTHON_OBJECT_FLAVORS: Dict["ModelVersion.ModelFlavor.V", ModelFlavor] = {
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_PYTORCH"): PyTorchModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_SKLEARN"): ScikitLearnModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_XGBOOST"): XGBoostModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_LIGHTGBM"): LightGBMModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_KERAS"): KerasModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_TENSORFLOW"): TensorFlowModelFlavor(),
    ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_CATBOOST"): CatBoostModelFlavor(),
    ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_HUGGINGFACE"
    ): HuggingFaceModelFlavor(),
}

# mypy proto enum typing issue https://github.com/protocolbuffers/protobuf/issues/8175
PYTHON_CLASS_NAME_TO_PROTO_FLAVORS: Dict[str, "ModelVersion.ModelFlavor.V"] = {
    PyTorchModelFlavor.__name__: ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_PYTORCH"),
    ScikitLearnModelFlavor.__name__: ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_SKLEARN"
    ),
    XGBoostModelFlavor.__name__: ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_XGBOOST"),
    LightGBMModelFlavor.__name__: ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_LIGHTGBM"
    ),
    KerasModelFlavor.__name__: ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_KERAS"),
    TensorFlowModelFlavor.__name__: ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_TENSORFLOW"
    ),
    CatBoostModelFlavor.__name__: ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_CATBOOST"
    ),
    HuggingFaceModelFlavor.__name__: ModelVersion.ModelFlavor.Value(
        "MODEL_FLAVOR_HUGGINGFACE"
    ),
}

# Order matters for matching
PYTHON_FLAVORS: List[ModelFlavor] = [
    # HF Flavor should come before Pytorch and Tensorflow since it uses Pytorch and Tensorflow as back bone
    HuggingFaceModelFlavor(),
    KerasModelFlavor(),
    PyTorchModelFlavor(),
    TensorFlowModelFlavor(),
    ScikitLearnModelFlavor(),
    XGBoostModelFlavor(),
    LightGBMModelFlavor(),
    CatBoostModelFlavor(),
]
