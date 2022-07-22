from typing import Optional
from unittest.mock import MagicMock

from layer.clients.logged_data_service import LoggedDataClient


def get_logged_data_service_client_with_mocks(
    logged_data_api_stub: Optional[MagicMock] = None,
) -> LoggedDataClient:
    logged_data_client = LoggedDataClient()
    logged_data_client._service = (  # pylint: disable=protected-access
        logged_data_api_stub if logged_data_api_stub is not None else MagicMock()
    )
    return logged_data_client
