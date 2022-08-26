from typing import Any

import requests  # type: ignore


class FileUploader:
    def __init__(self) -> None:
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[408, 429],  # timeout  # too many requests
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)

        self._session = requests.Session()
        self._session.mount("https://", adapter=adapter)

    def upload(self, url: str, data: Any = None) -> None:
        resp = self._session.put(url, data=data)
        resp.raise_for_status()

    def close(self) -> None:
        self._session.close()
