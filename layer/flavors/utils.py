from typing import Dict, List, Optional

from layerapi.api.entity.model_version_pb2 import (  # pylint: disable=unused-import
    ModelVersion,
)

from layer.contracts.models import TrainedModelObject
from layer.flavors.base import ModelFlavor

from .base import ModelFlavor  # noqa
from .catboost import CatBoostModelFlavor  # noqa
from .huggingface import HuggingFaceModelFlavor  # noqa
from .keras import KerasModelFlavor  # noqa
from .lightgbm import LightGBMModelFlavor  # noqa
from .pytorch import PyTorchModelFlavor  # noqa
from .sklearn import ScikitLearnModelFlavor  # noqa
from .tensorflow import TensorFlowModelFlavor  # noqa
from .xgboost import XGBoostModelFlavor  # noqa


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

# mypy proto enum typing issue https://github.com/protocolbuffers/protobuf/issues/8175
PYTHON_CLASS_NAME_TO_PROTO_FLAVORS: Dict[str, "ModelVersion.ModelFlavor"] = {
    flavor.__class__.__name__: flavor.PROTO_FLAVOR for flavor in PYTHON_FLAVORS
}

PROTO_TO_PYTHON_OBJECT_FLAVORS: Dict["ModelVersion.ModelFlavor", ModelFlavor] = {
    flavor.PROTO_FLAVOR: flavor for flavor in PYTHON_FLAVORS
}


def get_flavor_for_model(model_object: TrainedModelObject) -> Optional[ModelFlavor]:
    matching_flavor: Optional[ModelFlavor] = None
    for flavor in PYTHON_FLAVORS:
        if flavor.can_interpret_object(model_object):
            matching_flavor = flavor
            break
    return matching_flavor


def get_flavor_for_proto(
    proto_flavor: "ModelVersion.ModelFlavor",
) -> Optional[ModelFlavor]:
    return PROTO_TO_PYTHON_OBJECT_FLAVORS.get(proto_flavor)
