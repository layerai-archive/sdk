from typing import Awaitable, Callable, Union

from yarl import URL

from layer.config import DEFAULT_PATH, DEFAULT_URL, ConfigManager
from layer.exceptions.exceptions import (
    UserConfigurationError,
    UserNotLoggedInException,
    UserWithoutAccountError,
)
from layer.utils.async_utils import asyncio_run_in_thread

from .utils import sdk_function


@sdk_function
def login(url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param url: Not used.
    :raise UserNotLoggedInException: If the user is not logged in.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Logs user in to Layer. You might be prompted to enter an access token from a web page that is provided.

    .. code-block:: python

        layer.login()

    """

    async def _login(manager: ConfigManager, login_url: URL) -> None:
        await manager.login_headless(login_url)

    _refresh_and_login_if_needed(url, _login)


@sdk_function
def login_with_access_token(
    access_token: str, url: Union[URL, str] = DEFAULT_URL
) -> None:
    """
    :param access_token: A valid access token retrieved from Layer.
    :param url: Not used.
    :raise UserNotLoggedInException: If the user is not logged in.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Log in with an access token. You might be prompted to enter an access token from a web page that is provided.

    .. code-block:: python

        layer.login_with_access_token(TOKEN)
    """

    async def _login(manager: ConfigManager, login_url: URL) -> None:
        await manager.login_with_access_token(login_url, access_token)

    _refresh_and_login_if_needed(url, _login)


def _refresh_and_login_if_needed(
    url: Union[URL, str], login_func: Callable[[ConfigManager, URL], Awaitable[None]]
) -> None:
    async def _login(login_url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        try:
            config = await manager.refresh()
            if not config.is_guest and config.url == login_url:
                # no need to re-login
                return
        except (UserNotLoggedInException, UserConfigurationError):
            pass

        try:
            await login_func(manager, login_url)
        except UserWithoutAccountError as ex:
            raise SystemExit(ex)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login(url))


@sdk_function
def login_as_guest(url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param url: (optional) The target platform where the user wants to log in.

    Logs user in to Layer as a guest user. Guest users can view public projects, but they cannot add, edit, or delete entities.

    .. code-block:: python

        layer.login_as_guest() # Uses default target platform URL

    """

    async def _login_as_guest(url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        await manager.login_as_guest(url)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login_as_guest(url))


@sdk_function
def login_with_api_key(api_key: str, url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param access_token: A valid API key retrieved from Layer UI.
    :param url: Not used.
    :raise AuthException: If the API key is invalid.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Log in with an API key.

    .. code-block:: python

        layer.login_with_api_key(API_KEY)
    """

    async def _login(url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        await manager.login_with_api_key(url, api_key)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login(url))


@sdk_function
def logout() -> None:
    """
    Log out of Layer.

    .. code-block:: python

        layer.logout()

    """
    asyncio_run_in_thread(ConfigManager(DEFAULT_PATH).logout())


@sdk_function
def show_api_key() -> None:
    config = ConfigManager(DEFAULT_PATH).load()
    print(config.credentials.refresh_token)
