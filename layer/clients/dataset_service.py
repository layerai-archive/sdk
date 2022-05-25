import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas
import pyarrow
from layerapi.api.service.dataset.dataset_api_pb2 import Command
from layerapi.api.value.ticket_pb2 import PartitionTicket
from pyarrow import flight as fl
from pyarrow.lib import ArrowKeyError

from layer.cache.cache import Cache
from layer.utils.grpc import create_grpc_ssl_config


class DatasetClientError(Exception):
    pass


def _dataset_exception_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    def inner_function(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ArrowKeyError as e:  # pytype: disable=mro-error
            not_found_message = str(e).replace(
                "gRPC returned not found error, with message: ",
                "",
            )
            raise DatasetClientError(not_found_message)
        except DatasetClientError as e:
            raise e
        except Exception as e:
            raise DatasetClientError("Query failed") from e

    return inner_function


@dataclass(frozen=True)
class PartitionMetadata:
    location: str
    format: int = PartitionTicket.Format.FORMAT_PARQUET
    checksum: str = ""


class Partition:
    def __init__(
        self,
        reader: Union[fl.FlightStreamReader, pandas.DataFrame],
        from_cache: bool = False,
    ):
        self._reader = reader
        self._from_cache = from_cache

    def to_pandas(self) -> pandas.DataFrame:
        if isinstance(self._reader, pandas.DataFrame):
            return self._reader
        return self._reader.read_pandas()

    @property
    def from_cache(self) -> bool:
        return self._from_cache


class _FlightCallMetadataMiddleware(fl.ClientMiddleware):
    def __init__(self, call_metadata: List[Tuple[str, str]], client: "DatasetClient"):
        super().__init__()
        self._call_metadata = call_metadata
        self._client = client

    def sending_headers(self) -> Any:
        headers = {i[0]: i[1] for i in self._call_metadata}
        headers["x-request-id"] = str(uuid.uuid4())
        return headers


class _FlightCallMetadataMiddlewareFactory(fl.ClientMiddlewareFactory):
    def __init__(self, call_metadata: List[Tuple[str, str]], client: "DatasetClient"):
        super().__init__()
        self._call_metadata = call_metadata
        self._client = client

    def start_call(self, info: fl.CallInfo) -> Any:
        return _FlightCallMetadataMiddleware(self._call_metadata, self._client)


class DatasetClient:
    def __init__(self, address_and_port: str, access_token: str) -> None:
        self._endpoint = f"grpc+tls://{address_and_port}"
        self._ssl_config = create_grpc_ssl_config(
            address_and_port, do_verify_ssl=True, do_force_cadata_load=True
        )
        self._middleware = [
            _FlightCallMetadataMiddlewareFactory(
                [("authorization", f"Bearer {access_token}")], self
            )
        ]
        self._flight = fl.FlightClient(
            self._endpoint,
            override_hostname=f"dataset.{address_and_port.partition(':')[0]}",
            tls_root_certs=self._ssl_config.cadata,
            middleware=self._middleware,
        )

    def health_check(self) -> str:
        buf = pyarrow.allocate_buffer(0)
        action = fl.Action("HealthCheck", buf)
        result = next(self._flight.do_action(action))
        return result.body.to_pybytes().decode("utf-8")

    @_dataset_exception_handler
    def get_partitions_metadata(self, command: Command) -> List[PartitionMetadata]:
        descriptor = fl.FlightDescriptor.for_command(command.SerializeToString())
        flight_info = self._flight.get_flight_info(descriptor)
        partitions_metadata = []
        for endpoint in flight_info.endpoints:
            ticket = PartitionTicket()
            ticket.ParseFromString(endpoint.ticket.ticket)
            for location in endpoint.locations:
                partitions_metadata.append(
                    PartitionMetadata(
                        location=location.uri.decode("utf-8"),
                        format=ticket.format,
                        checksum=ticket.checksum,
                    )
                )
        return partitions_metadata

    @_dataset_exception_handler
    def fetch_partition(
        self,
        partition_metadata: PartitionMetadata,
        no_cache: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> Partition:
        # support only parquet partitions
        if partition_metadata.format != PartitionTicket.Format.FORMAT_PARQUET:
            raise DatasetClientError(
                f"Unsupported partition format: {partition_metadata.format}"
            )
        # caching only possible if checksum available
        if len(partition_metadata.checksum) > 0 and not no_cache:
            cache = Cache(cache_dir=cache_dir).initialise()
            cache_path = cache.get_path_entry(partition_metadata.checksum)
            from_cache = cache_path is not None
            if not from_cache:
                download_path = cache.cache_dir.joinpath(str(uuid.uuid4()))
                urllib.request.urlretrieve(  # nosec urllib_urlopen
                    partition_metadata.location, filename=download_path
                )
                cache_path = cache.put_path_entry(
                    partition_metadata.checksum, download_path
                )
            return Partition(_read_parquet(cache_path), from_cache=from_cache)  # type: ignore
        # read directly from the remote location
        return Partition(_read_parquet(partition_metadata.location))

    @_dataset_exception_handler
    def get_dataset_writer(self, command: Command, schema: Any) -> Any:
        descriptor = fl.FlightDescriptor.for_command(command.SerializeToString())
        return self._flight.do_put(descriptor, schema)


def _read_parquet(path: Union[str, Path]) -> pandas.DataFrame:
    return pandas.read_parquet(path, engine="pyarrow")
