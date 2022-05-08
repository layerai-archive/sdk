# flake8: noqa
from pathlib import Path

import layer
from layer import DerivedDataset, Model
from layer.annotation_processor import (
    parse_entity_dependencies,
    parse_entity_dependencies_from_source_code,
)
from layer.client import Dataset, DerivedDataset


class TestAnnotationProcessor:
    def test_single_parameter_single_argument(self) -> None:
        def fut(non_related, a: layer.DerivedDataset("aa")):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

    def test_single_parameter_single_argument_without_module_name(self) -> None:
        def fut(non_related, a: DerivedDataset("aa")):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

    def test_single_parameter_multiple_arguments(self) -> None:
        def fut(non_related, a: layer.DerivedDataset("xx")):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "xx"

    def test_multiple_parameter_multiple_arguments(self) -> None:
        def fut(
            non_related,
            a: layer.DerivedDataset("first"),
            b: layer.DerivedDataset("second"),
            c: layer.Dataset("yy"),
            d: layer.Model("zz"),
        ):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "first"

        assert isinstance(annotations.derived_datasets["b"], DerivedDataset)
        assert annotations.derived_datasets["b"].name == "second"

        assert isinstance(annotations.raw_datasets["c"], Dataset)
        assert annotations.raw_datasets["c"].name == "yy"

        assert isinstance(annotations.models["d"], Model)
        assert annotations.models["d"].name == "zz"

    def test_multiple_parameter_with_unrelated_type_and_multiple_arguments(
        self,
    ) -> None:
        def fut(
            path: Path,
            a: layer.DerivedDataset("first"),
            b: layer.DerivedDataset("second"),
        ):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "first"

        assert isinstance(annotations.derived_datasets["b"], DerivedDataset)
        assert annotations.derived_datasets["b"].name == "second"

    def test_multiple_parameters_different_types(self) -> None:
        def fut(non_related, a: layer.DerivedDataset("aa"), b: layer.Dataset("bb")):
            pass

        annotations = parse_entity_dependencies(fut)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

        assert isinstance(annotations.raw_datasets["b"], Dataset)
        assert annotations.raw_datasets["b"].name == "bb"

    def test_parse_entity_dependencies_from_source_code_fully_qualified_names(
        self,
    ) -> None:
        source = """
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from layer import DerivedDataset, Train

def train_model(non_related, a: layer.DerivedDataset("aa"), b: layer.DerivedDataset("bb"), c: layer.Model("cc")) -> Any:
    pass
                """

        annotations = parse_entity_dependencies_from_source_code(source)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

        assert isinstance(annotations.derived_datasets["b"], DerivedDataset)
        assert annotations.derived_datasets["b"].name == "bb"

        assert isinstance(annotations.models["c"], Model)
        assert annotations.models["c"].name == "cc"

    def test_parse_entity_dependencies_from_source_code(self) -> None:
        source = """
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from layer import DerivedDataset, Train

def train_model(non_related, a: DerivedDataset("aa"), b: DerivedDataset("bb"), c: Model("cc")) -> Any:
    pass
                """

        annotations = parse_entity_dependencies_from_source_code(source)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

        assert isinstance(annotations.derived_datasets["b"], DerivedDataset)
        assert annotations.derived_datasets["b"].name == "bb"

        assert isinstance(annotations.models["c"], Model)
        assert annotations.models["c"].name == "cc"

    def test_parse_entity_dependencies_with_versions_from_source_code(self) -> None:
        source = """
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from layer import DerivedDataset, Train

def train_model(a: DerivedDataset("aa"), b: DerivedDataset("bb:1"), c: DerivedDataset("cc:2.3"), d: DerivedDataset("dd:my_tag")) -> Any:
    pass
                """

        annotations = parse_entity_dependencies_from_source_code(source)
        assert isinstance(annotations.derived_datasets["a"], DerivedDataset)
        assert annotations.derived_datasets["a"].name == "aa"

        assert isinstance(annotations.derived_datasets["b"], DerivedDataset)
        assert annotations.derived_datasets["b"].name == "bb"

        assert isinstance(annotations.derived_datasets["c"], DerivedDataset)
        assert annotations.derived_datasets["c"].name == "cc"

        assert isinstance(annotations.derived_datasets["d"], DerivedDataset)
        assert annotations.derived_datasets["d"].name == "dd"
