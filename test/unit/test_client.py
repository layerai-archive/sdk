import logging

from layer.data_classes import Dataset


logger = logging.getLogger(__name__)


def test_dataset_to_pandas_returns_default_empty_dataframe() -> None:
    assert len(Dataset("test_dataset").to_pandas()) == 0
