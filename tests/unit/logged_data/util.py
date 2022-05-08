import logging
from typing import Optional
from unittest.mock import MagicMock

from layer.config import ClientConfig
from layer.logged_data.client import LoggedDataClient


def get_logged_data_service_client_with_mocks(
    logged_data_api_stub: Optional[MagicMock] = None,
) -> LoggedDataClient:
    config_mock = MagicMock(spec=ClientConfig)
    logged_data_client = LoggedDataClient(
        config=config_mock, logger=MagicMock(spec_set=logging.getLogger())
    )
    logged_data_client._service = (
        logged_data_api_stub if logged_data_api_stub is not None else MagicMock()
    )
    return logged_data_client
