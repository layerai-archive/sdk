import os
from unittest.mock import patch

import layer
from layer.decorators.dataset_decorator import dataset
from layer.decorators.model_decorator import model


def test_remote_run_executables_feature_flag_arg_passed():
    with patch("layer.executables.runner.remote_run") as remote_run:
        layer.run(functions=[d, m], executables_feature=True)
        remote_run.assert_called_once_with([d, m])


def test_remote_run_executables_feature_flag_env_var():
    with patch.dict(os.environ, {"LAYER_EXECUTABLES": "1"}), patch(
        "layer.executables.runner.remote_run"
    ) as remote_run:
        layer.run(functions=[d, m])
        remote_run.assert_called_once_with([d, m])


@dataset("d")
def d():
    return [1]


@model("m")
def m():
    return [2]
