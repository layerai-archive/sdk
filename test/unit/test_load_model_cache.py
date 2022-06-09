import logging
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor
from yarl import URL

from layer.clients.model_catalog import ModelCatalogClient
from layer.config.config import ClientConfig
from layer.contracts.models import Model
from layer.contracts.tracker import ResourceTransferState
from layer.flavors.base import ModelFlavor
from layer.types import ModelObject


logger = logging.getLogger(__name__)


def test_model_load_from_cache(tmp_path: Path) -> None:
    client_config: ClientConfig = Mock(
        s3=Mock(
            endpoint_url=URL("https://model.catalog"),
        )
    )

    model_catalog = ModelCatalogClient(client_config, logger, cache_dir=tmp_path)

    model_flavor = DummyModelFlavor()
    model: Model = Mock(
        id=uuid.UUID("a7c02598-7910-4f96-b8e6-bc6b62bd214f"),
        name="model_definition",
        flavor=model_flavor,
    )

    model_cache_dir: Path = tmp_path / "cache" / str(model.id)

    with patch("layer.utils.s3.S3Util.download_dir") as s3_download:
        s3_download.side_effect = download_model_files_from_s3
        model_catalog.load_model_runtime_objects(model, state=ResourceTransferState())
        # assert model was downloaded and not loaded from the cache dir
        s3_download.assert_called_once()
        model_flavor.model_impl.assert_called_once_with(model_cache_dir.as_uri())

    model_flavor.model_impl.reset_mock()

    with patch("layer.utils.s3.S3Util.download_dir") as s3_download:
        model_catalog.load_model_runtime_objects(model, state=ResourceTransferState())
        # assert model was loaded from the cache dir without downloading
        s3_download.assert_not_called()
        model_flavor.model_impl.assert_called_once_with(model_cache_dir.as_uri())


def test_model_load_from_cache_does_not_cache_if_no_cache_true(tmp_path: Path) -> None:
    client_config: ClientConfig = Mock(
        s3=Mock(
            endpoint_url=URL("https://model.catalog"),
        )
    )

    model_catalog = ModelCatalogClient(client_config, logger, cache_dir=tmp_path)

    model_flavor = DummyModelFlavor()
    model: Model = Mock(
        id=uuid.UUID("0b4d4ef8-c81e-493c-bf61-9e437a8c6f6e"),
        name="model_definition",
        flavor=model_flavor,
    )

    with patch("layer.utils.s3.S3Util.download_dir") as s3_download:
        s3_download.side_effect = download_model_files_from_s3
        model_catalog.load_model_runtime_objects(
            model, state=ResourceTransferState(), no_cache=True
        )
        model_download_dir = s3_download.call_args[1]["local_dir"]
        # assert model was downloaded and not loaded from the cache dir
        s3_download.assert_called_once()
        model_flavor.model_impl.assert_called_once_with(model_download_dir.as_uri())


def download_model_files_from_s3(*args, **kwargs) -> None:
    # mock the download from s3 by creating the download directory
    local_dir: Path = kwargs["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=False)


class DummyModelFlavor(ModelFlavor):
    MODULE_KEYWORD = "dummy"
    PROTO_FLAVOR = PbModelFlavor.Value("MODEL_FLAVOR_INVALID")

    def __init__(self) -> None:
        super().__init__()
        self.model_impl = Mock(name="dummy_model_impl")

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
    ) -> None:
        return

    def load_model_from_directory(self, directory: Path) -> ModelObject:
        return self.model_impl(directory.as_uri())
