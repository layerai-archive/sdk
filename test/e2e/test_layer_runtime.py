from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import layer
from layer import dataset
from layer.contracts.projects import Project
from layer.executables.function import Function
from layer.executables.layer_runtime import LayerFunctionRuntime


def test_dataset_build(initialized_project: Project, tmpdir: Path):
    @dataset("test_dataset")
    def build_dataset():
        return pd.DataFrame({"a": [1, 2, 3]})

    function = Function.from_decorated(build_dataset)
    executable_path = function.package(output_dir=tmpdir)
    LayerFunctionRuntime.execute(executable_path, project=initialized_project.name)

    actual = layer.get_dataset("test_dataset").to_pandas()

    assert_frame_equal(actual, pd.DataFrame({"a": [1, 2, 3]}))
