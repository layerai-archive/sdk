import contextlib
from unittest.mock import MagicMock, patch

import pandas
import torch
from pyarrow.flight import FlightStreamReader

from layer import get_dataset
from layer.api.service.dataset.dataset_api_pb2 import Command, DatasetQuery
from layer.api.value.ticket_pb2 import DatasetPathTicket, DataTicket, PartitionTicket
from layer.client import DataCatalogClient
from layer.common import LayerClient
from layer.config import ClientConfig, Config
from layer.dataset_client import DatasetClient, Partition, PartitionMetadata


def test_get_dataset_to_pandas_calls_dataset_api(test_project_name):
    dataset_client = MagicMock(spec_set=DatasetClient)
    with _dataset_client_mock(dataset_client_mock=dataset_client):
        get_dataset("dataset_42").to_pandas()

    expected_ticket = DataTicket(
        dataset_path_ticket=DatasetPathTicket(
            path=f"{test_project_name}/datasets/dataset_42"
        )
    )
    expected_command = Command(dataset_query=DatasetQuery(ticket=expected_ticket))
    dataset_client.get_partitions_metadata.assert_called_with(expected_command)


def test_get_dataset_to_pandas_calls_dataset_api_with_path():
    dataset_client = MagicMock(spec_set=DatasetClient)
    dataset_path = "another-project/datasets/dataset_42"
    with _dataset_client_mock(dataset_client_mock=dataset_client):
        get_dataset(dataset_path).to_pandas()

    expected_ticket = DataTicket(
        dataset_path_ticket=DatasetPathTicket(path=dataset_path)
    )
    expected_command = Command(dataset_query=DatasetQuery(ticket=expected_ticket))
    dataset_client.get_partitions_metadata.assert_called_with(expected_command)


def test_get_dataset_to_pandas_returns_empty_data_frame_for_no_data():
    with _dataset_client_mock():
        assert get_dataset("dataset_42").to_pandas().empty


def test_get_dataset_to_pytorch_returns_pytorch_dataloader():
    def transform(row):
        return row.x, row.y + "!"

    df = pandas.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    with patch("layer.client.Dataset.to_pandas", return_value=df):
        with _dataset_client_mock():
            ds = get_dataset("dummy").to_pytorch(transform)
            item, (x, y) = next(enumerate(ds))
            assert x == torch.tensor([1])
            assert y == ("a!",)
            assert len(ds) == 2


class _MockFlightStreamReader(FlightStreamReader):
    def __init__(self, data_frame: pandas.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_frame = data_frame

    def read_pandas(self, **options) -> pandas.DataFrame:
        return self._data_frame


def test_get_dataset_to_pandas_concatenates_the_result_from_multiple_partitions():
    dataset_client = MagicMock(spec_set=DatasetClient)
    partition_metadata_1 = PartitionMetadata(
        "grpc+tls://address1", format=PartitionTicket.Format.FORMAT_NATIVE
    )
    partition_metadata_2 = PartitionMetadata(
        "grpc+tls://address2", format=PartitionTicket.Format.FORMAT_NATIVE
    )
    dataset_client.get_partitions_metadata.return_value = [
        partition_metadata_1,
        partition_metadata_2,
    ]

    partitions = {
        partition_metadata_1: Partition(
            _MockFlightStreamReader(pandas.DataFrame({"x": [1, 2]}))
        ),
        partition_metadata_2: Partition(
            _MockFlightStreamReader(pandas.DataFrame({"x": [3, 4, 6, 6, 7]}))
        ),
    }

    def fetch_partition(partition_metadata, no_cache=False):
        return partitions[partition_metadata]

    dataset_client.fetch_partition.side_effect = fetch_partition
    with _dataset_client_mock(dataset_client_mock=dataset_client):
        dataset_data_frame = get_dataset("dataset_43").to_pandas()
        assert dataset_data_frame.equals(pandas.DataFrame({"x": [1, 2, 3, 4, 6, 6, 7]}))


@contextlib.contextmanager
def _dataset_client_mock(dataset_client_mock=None):
    dataset_client = dataset_client_mock or MagicMock(spec_set=DatasetClient)

    data_catalog_client = DataCatalogClient(
        config=MagicMock(),
        logger=MagicMock(),
        flight_client=MagicMock(),
        dataset_client=dataset_client,
    )

    client = MagicMock()
    client.__enter__.return_value = MagicMock(
        set_spec=LayerClient, data_catalog=data_catalog_client
    )

    config = MagicMock(
        set_spec=Config,
        client=MagicMock(set_spec=ClientConfig, grpc_gateway_address="grpc.test"),
    )

    async def config_refresh(*, allow_guest: bool = False):
        return config

    with patch("layer.main.LayerClient.init", return_value=client), patch(
        "layer.config.ConfigManager.refresh", side_effect=config_refresh
    ):
        yield
