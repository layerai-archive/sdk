from typing import Union

import pytest
from yarl import URL

from layer import Context
from layer.context import get_active_context
from layer.contracts.asset import AssetType
from layer.contracts.datasets import DatasetBuild
from layer.contracts.project_full_name import ProjectFullName
from layer.training.base_train import BaseTrain


EXAMPLE_LAYER_URL = URL("https://app.layer.ai")


EXAMPLE_PROJECT = ProjectFullName(
    account_name="acc",
    project_name="proj",
)


class ExampleContextHolder:
    def __init__(self, context: Context):
        self._context = context

    def context(self) -> Context:
        return self._context


class TestContext:
    def test_correct_context_returned(self) -> None:
        assert get_active_context() is None
        with Context(
            url=EXAMPLE_LAYER_URL,
            project_full_name=EXAMPLE_PROJECT,
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        ) as ctx:
            assert get_active_context() == ctx

        assert get_active_context() is None

    def test_context_reference_works_after_context_exits(self) -> None:
        assert get_active_context() is None
        with Context(
            url=EXAMPLE_LAYER_URL,
            project_full_name=EXAMPLE_PROJECT,
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        ) as ctx:
            holder = ExampleContextHolder(context=ctx)
            assert get_active_context() == ctx

        assert holder.context() is not None
        assert get_active_context() is None

    class FakeTrain(BaseTrain):
        def __init__(self, version: str, index: int):
            self._version = version
            self._train_index = index

        def get_version(self) -> str:
            return self._version

        def get_train_index(self) -> str:
            return str(self._train_index)

    @pytest.mark.parametrize(
        ("asset_name", "asset_type", "build_or_train", "expected_url"),
        [
            (
                "the-model",
                AssetType.MODEL,
                None,
                URL(f"{EXAMPLE_LAYER_URL}/{EXAMPLE_PROJECT.path}/models/the-model"),
            ),
            (
                "the-model",
                AssetType.MODEL,
                FakeTrain("1", 34),
                URL(
                    f"{EXAMPLE_LAYER_URL}/{EXAMPLE_PROJECT.path}/models/the-model?v=1.34"
                ),
            ),
            (
                "the-dataset",
                AssetType.DATASET,
                None,
                URL(f"{EXAMPLE_LAYER_URL}/{EXAMPLE_PROJECT.path}/datasets/the-dataset"),
            ),
        ],
    )
    def test_context_url_returns_full_url(
        self,
        asset_name: str,
        asset_type: AssetType,
        build_or_train: Union[DatasetBuild, BaseTrain],
        expected_url: URL,
    ) -> None:
        with Context(
            url=EXAMPLE_LAYER_URL,
            project_full_name=EXAMPLE_PROJECT,
            dataset_build=build_or_train
            if isinstance(build_or_train, DatasetBuild)
            else None,
            train=build_or_train if isinstance(build_or_train, BaseTrain) else None,
            asset_name=asset_name,
            asset_type=asset_type,
        ) as ctx:
            assert ctx.url() == expected_url
