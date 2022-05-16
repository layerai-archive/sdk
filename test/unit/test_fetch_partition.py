from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from layer.api.value.ticket_pb2 import PartitionTicket
from layer.clients.dataset_client import (
    DatasetClient,
    DatasetClientError,
    PartitionMetadata,
)


def test_fetch_partition_throws_format_other_than_parquet(
    dataset_client: DatasetClient,
):
    part_meta = PartitionMetadata(
        "http://localhost", format=PartitionTicket.FORMAT_NATIVE
    )
    with pytest.raises(DatasetClientError, match="Unsupported partition format: 1"):
        dataset_client.fetch_partition(part_meta)


def test_fetch_partition_reads_parquet_from_remote_location_when_checksum_unavailable(
    dataset_client: DatasetClient,
):
    part_meta = PartitionMetadata(location="http://localhost/datasets/42")
    with patch("pandas.read_parquet") as read_parquet:
        dataset_client.fetch_partition(part_meta)
        read_parquet.assert_called_once_with(
            "http://localhost/datasets/42", engine="pyarrow"
        )


def test_fetch_partition_fetches_partition_to_cache_dir(
    dataset_client: DatasetClient, tmp_path
):
    part_meta = PartitionMetadata(
        location="http://localhost/datasets/1",
        checksum="69fe17788c2e684f4a78f4d242e77dc9",
    )
    with patch("pandas.read_parquet"), patch(
        "urllib.request.urlretrieve",
    ) as urlretrieve:
        dataset_client.fetch_partition(part_meta, cache_dir=tmp_path)
        _, kwargs = urlretrieve.call_args
        partition_file = kwargs["filename"]
        assert partition_file.relative_to(tmp_path / "cache")
        assert partition_file.is_file


def test_fetch_partition_fetches_and_caches(dataset_client: DatasetClient, tmp_path):
    part_checksum = "69fe17788c2e684f4a78f4d242e77dc9"
    part_meta = PartitionMetadata(
        location="http://localhost/datasets/1",
        checksum=part_checksum,
    )
    part_cache_path = tmp_path / "cache" / part_checksum
    # fetch from the remote location and cache
    with patch("pandas.read_parquet") as read_parquet, patch(
        "urllib.request.urlretrieve",
    ) as urlretrieve:
        urlretrieve.side_effect = urlretrieve_side_effect
        partition = dataset_client.fetch_partition(part_meta, cache_dir=tmp_path)
        urlretrieve.assert_called_once_with(
            "http://localhost/datasets/1", filename=mock.ANY
        )
        part_download_file = urlretrieve.call_args[1]["filename"]
        assert part_download_file.parent == tmp_path / "cache"
        read_parquet.assert_called_once_with(part_cache_path, engine="pyarrow")
        assert not partition.from_cache
        assert part_cache_path.exists()
    # fetch again, this time from the cache
    with patch("pandas.read_parquet") as read_parquet, patch(
        "urllib.request.urlretrieve",
    ) as urlretrieve:
        partition = dataset_client.fetch_partition(part_meta, cache_dir=tmp_path)
        urlretrieve.assert_not_called
        read_parquet.assert_called_once_with(part_cache_path, engine="pyarrow")
        assert (
            partition.from_cache
        )  # this time partition should be retrieved from the cache


def test_fetch_partition_ignores_cache_when_no_cache_set(
    dataset_client: DatasetClient, tmp_path
):
    part_checksum = "0111b73506a99f506dd7c76434215d77"
    part_meta = PartitionMetadata(
        location="http://localhost/datasets/1",
        checksum=part_checksum,
    )
    part_cache_path = tmp_path / "cache" / part_checksum
    # fetch from the remote location and cache
    with patch("pandas.read_parquet") as read_parquet, patch(
        "urllib.request.urlretrieve",
    ) as urlretrieve:
        urlretrieve.side_effect = urlretrieve_side_effect
        partition = dataset_client.fetch_partition(part_meta, cache_dir=tmp_path)
        urlretrieve.assert_called_once_with(
            "http://localhost/datasets/1", filename=mock.ANY
        )
        part_download_file = urlretrieve.call_args[1]["filename"]
        assert part_download_file.parent == tmp_path / "cache"
        read_parquet.assert_called_once_with(part_cache_path, engine="pyarrow")
        assert not partition.from_cache
        assert part_cache_path.exists()
    # force fetch from the remote location
    with patch("pandas.read_parquet") as read_parquet, patch(
        "urllib.request.urlretrieve",
    ) as urlretrieve:
        partition = dataset_client.fetch_partition(
            part_meta, cache_dir=tmp_path, no_cache=True
        )
        urlretrieve.assert_not_called
        read_parquet.assert_called_once_with(
            "http://localhost/datasets/1", engine="pyarrow"
        )
        assert not partition.from_cache


def urlretrieve_side_effect(*args, **kwargs):
    Path(kwargs["filename"]).touch()


@pytest.fixture()
def dataset_client() -> DatasetClient:
    return DatasetClient(address_and_port="localhost:8080", access_token="access_token")
