from dataclasses import replace
from pathlib import Path
from typing import Optional, Type

from yarl import URL

from layer.auth import (
    CodeClient,
    CredentialsClient,
    HeadlessCodeClient,
    WebBrowserCodeClient,
)
from layer.config import DEFAULT_PATH, DEFAULT_URL, Config, ConfigStore, Credentials
from layer.exceptions.exceptions import (
    InvalidConfigurationError,
    MissingConfigurationError,
    UserAccessTokenExpiredError,
    UserConfigurationError,
    UserNotLoggedInException,
    UserWithoutAccountError,
)

from .config_client import ConfigClient
from .guest_login_client import get_guest_auth_token


class ConfigManager:
    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._get_default_path()
        assert path
        self._store = ConfigStore(path)

    @staticmethod
    def _get_default_path() -> Path:
        return DEFAULT_PATH

    def load(self) -> Config:
        return self._store.load()

    async def refresh(self, *, allow_guest: bool = False) -> Config:
        from aiohttp import ClientSession

        try:
            config = self._store.load()
        except MissingConfigurationError:
            if allow_guest:
                config = await self.login_as_guest(DEFAULT_URL)
            else:
                await self._logout()
                raise UserNotLoggedInException()

        except InvalidConfigurationError as cex:
            await self._logout()
            raise UserConfigurationError(cex.path)

        if not config.auth.is_enabled or not config.credentials.is_access_token_expired:
            return config

        async with ClientSession() as client:
            creds_client = CredentialsClient(
                client=client,
                url=config.auth.token_url,
                client_id=config.auth.client_id,
            )
            creds = await creds_client.refresh(config.credentials)

        config = config.with_credentials(creds)
        self._store.save(config)
        return config

    async def _login(
        self, url: URL, code_client_factory: Type[CodeClient] = HeadlessCodeClient
    ) -> Config:
        from aiohttp import ClientSession

        await self._logout()

        async with ClientSession() as client:
            config_client = ConfigClient(client=client, url=url)
            config = await config_client.get_config()

            code_client = code_client_factory(  # pytype: disable=not-instantiable
                config=config.auth
            )
            code = await code_client.request()

            creds_client = CredentialsClient(
                client=client,
                url=config.auth.token_url,
                client_id=config.auth.client_id,
            )
            creds = await creds_client.request(code)

            if creds.is_authenticated_outside_organization:
                raise UserWithoutAccountError(config.url)

        config = config.with_credentials(creds)
        self._store.save(config)
        print(f"Successfully logged into {url}")
        return config

    async def _logout(self) -> None:
        self._store.delete()

    async def login(self, url: URL) -> Config:
        return await self._login(url, code_client_factory=WebBrowserCodeClient)

    async def login_as_guest(self, url: URL) -> Config:
        _url = url.with_host(f"grpc.{url.host}")
        grpc_url = f"{_url.host}:{_url.port}"

        token = get_guest_auth_token(str(grpc_url))

        return await self.login_with_access_token(url, token, config_is_guest=True)

    async def login_headless(self, url: URL) -> Config:
        return await self._login(url, code_client_factory=HeadlessCodeClient)

    async def login_with_access_token(
        self, url: URL, access_token: str, config_is_guest: bool = False
    ) -> Config:
        from aiohttp import ClientSession

        await self._logout()

        async with ClientSession() as client:
            config_client = ConfigClient(client=client, url=url)
            config = await config_client.get_config()

        creds = Credentials(access_token=access_token, refresh_token="")
        if config.auth.is_enabled:
            if creds.is_access_token_expired:
                raise UserAccessTokenExpiredError()

        config = config.with_credentials(creds)
        config = replace(config, is_guest=config_is_guest)
        self._store.save(config)
        if config_is_guest:
            print(f"Successfully logged into {url} as guest")
        else:
            print(f"Successfully logged into {url}")
        return config

    async def login_with_api_key(self, url: URL, api_key: str) -> Config:
        await self._logout()

        from aiohttp import ClientSession

        async with ClientSession() as client:
            config_client = ConfigClient(client=client, url=url)
            config = await config_client.get_config()

        creds = Credentials(access_token="", refresh_token=api_key)
        config = config.with_credentials(creds)

        self._store.save(config)
        await self.refresh()
        print(f"Successfully logged into {url}")
        return config

    async def logout(self) -> None:
        msg = "Logged out."
        try:
            self._store.load()
        except MissingConfigurationError:
            msg = "No active sessions."
        except InvalidConfigurationError:
            pass

        await self._logout()
        print(msg)
