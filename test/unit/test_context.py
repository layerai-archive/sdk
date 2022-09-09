import uuid
from typing import Union
from unittest.mock import MagicMock

import pytest
from yarl import URL

from layer import Context
from layer.context import get_active_context
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import DatasetBuild
from layer.contracts.models import ModelTrain, ModelTrainStatus
from layer.contracts.project_full_name import ProjectFullName


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
        asset_path = AssetPath(
            account_name=EXAMPLE_PROJECT.account_name,
            project_name=EXAMPLE_PROJECT.project_name,
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        )
        with Context(
            url=EXAMPLE_LAYER_URL,
            asset_path=asset_path,
        ) as ctx:
            assert get_active_context() == ctx

        assert get_active_context() is None

    def test_context_reference_works_after_context_exits(self) -> None:
        assert get_active_context() is None
        asset_path = AssetPath(
            account_name=EXAMPLE_PROJECT.account_name,
            project_name=EXAMPLE_PROJECT.project_name,
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        )
        with Context(
            url=EXAMPLE_LAYER_URL,
            asset_path=asset_path,
        ) as ctx:
            holder = ExampleContextHolder(context=ctx)
            assert get_active_context() == ctx

        assert holder.context() is not None
        assert get_active_context() is None

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
                ModelTrain(uuid.uuid4(), 34, ModelTrainStatus.IN_PROGRESS, "1.2"),
                URL(
                    f"{EXAMPLE_LAYER_URL}/{EXAMPLE_PROJECT.path}/models/the-model?v=1.2"
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
        build_or_train: Union[DatasetBuild, ModelTrain],
        expected_url: URL,
    ) -> None:
        asset_path = AssetPath(
            account_name=EXAMPLE_PROJECT.account_name,
            project_name=EXAMPLE_PROJECT.project_name,
            asset_name=asset_name,
            asset_type=asset_type,
        )
        with Context(
            url=EXAMPLE_LAYER_URL,
            asset_path=asset_path,
            client=MagicMock(),
            dataset_build=build_or_train
            if isinstance(build_or_train, DatasetBuild)
            else None,
            model_train=build_or_train
            if isinstance(build_or_train, ModelTrain)
            else None,
        ) as ctx:
            assert ctx.url() == expected_url
