import logging
from unittest.mock import MagicMock, call

import layer
from layer import Context, Dataset, Model
from layer.global_context import set_current_project_name
from layer.training.runtime.dependency_injector import (
    inject_annotated_dependencies,  # type: ignore
)


logger = logging.getLogger(__name__)


class TestInjector:
    def test_inject(self) -> None:
        # given
        def fut(
            non_related: str,
            c: layer.Model(  # type: ignore
                "other-org/other-project/models/third"  # noqa
            ),  # noqa  # type: ignore
            d: layer.DerivedDataset(  # type: ignore
                "other-project/datasets/fourth"  # noqa
            ),  # noqa  # type: ignore
            e: layer.Dataset("fifth"),  # type: ignore  # noqa  # noqa  # type: ignore
        ) -> None:
            pass

        model_factory = MagicMock(side_effect=lambda name: Model(name))
        dataset_factory = MagicMock(side_effect=lambda name: Dataset(name))

        # and
        set_current_project_name("current-project")

        inject_annotated_dependencies(
            fut,
            Context(),
            logger,
            dataset_factory=dataset_factory,
            model_factory=model_factory,
        )

        model_factory.assert_called_once_with(
            name="other-org/other-project/models/third"
        )
        dataset_factory.assert_has_calls(
            [
                call(name=layer.current_project_name() + "/datasets/fifth"),
                call(name="other-project/datasets/fourth"),
            ]
        )
