import asyncio
from dataclasses import replace
from typing import AsyncIterator, Optional
from unittest import mock
from urllib.parse import parse_qsl

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import unused_port
from aiohttp.web import (
    Application,
    HTTPBadRequest,
    HTTPException,
    HTTPForbidden,
    HTTPFound,
    HTTPOk,
    Request,
    Response,
    json_response,
)
from yarl import URL

from layer.auth import (
    AuthException,
    Code,
    CodeChallenge,
    CredentialsClient,
    HeadlessCodeClient,
    WebBrowserCodeClient,
    create_app_server,
    create_app_server_once,
    create_auth_code_app,
)
from layer.config import AuthConfig, Credentials
from layer.utils.async_utils import asyncio_run_in_thread


pytestmark = pytest.mark.asyncio


class _TestAuthHandler:
    def __init__(self, client_id: str) -> None:
        self._client_id = client_id

        self._code = "test_code"
        self._access_token = "test_access_token"
        self._access_token_refreshed = "test_access_token_refreshed"
        self._refresh_token = "test_refresh_token"
        self._access_token_expires_in = 1234

        self.raise_on_token: Optional[HTTPException] = None

    async def handle_authorize(self, request: Request) -> Response:
        url = URL(request.query["redirect_uri"]).with_query(code=self._code)
        raise HTTPFound(url)

    async def handle_token(self, request: Request) -> Response:
        if self.raise_on_token:
            raise self.raise_on_token
        assert request.headers["accept"] == "application/json"
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        payload = dict(parse_qsl(await request.text()))
        grant_type = payload["grant_type"]
        if grant_type == "authorization_code":
            assert payload == {
                "grant_type": "authorization_code",
                "code_verifier": mock.ANY,
                "code": self._code,
                "client_id": self._client_id,
                "redirect_uri": mock.ANY,
            }
            resp_payload = {
                "access_token": self._access_token,
                "expires_in": self._access_token_expires_in,
                "refresh_token": self._refresh_token,
            }
        else:
            assert payload == {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
            }
            resp_payload = {
                "access_token": self._access_token_refreshed,
                "expires_in": self._access_token_expires_in,
            }
        return json_response(resp_payload)


@pytest.fixture()
def auth_client_id() -> str:
    return "test_client_id"


@pytest.fixture()
def auth_handler(auth_client_id: str) -> _TestAuthHandler:
    return _TestAuthHandler(client_id=auth_client_id)


@pytest.fixture()
async def auth_server(auth_handler: _TestAuthHandler) -> AsyncIterator[URL]:
    app = Application()
    app.router.add_get("/authorize", auth_handler.handle_authorize)
    app.router.add_post("/oauth/token", auth_handler.handle_token)
    async with create_app_server_once(app, host="0.0.0.0", port=unused_port()) as url:
        yield url


@pytest.fixture()
async def auth_config(auth_client_id: str, auth_server: URL) -> AuthConfig:
    port = unused_port()
    return AuthConfig(
        auth_url=auth_server / "authorize",
        token_url=auth_server / "oauth" / "token",
        logout_url=auth_server / "v2" / "logout",
        client_id=auth_client_id,
        audience="https://development.layer.co",
        headless_callback_url=URL("https://development.layer.co/oauth/code"),
        callback_urls=[URL(f"http://127.0.0.1:{port}")],
        success_redirect_url=URL("https://development.layer.co"),
        failure_redirect_url=URL("https://development.layer.co/oauth/code"),
    )


class TestCredentialsClient:
    async def test_request(self, auth_config: AuthConfig) -> None:
        code = Code(CodeChallenge.create(), "test_code", auth_config.callback_urls[0])

        async with ClientSession() as client:
            creds_client = CredentialsClient(
                client=client,
                url=auth_config.token_url,
                client_id=auth_config.client_id,
            )
            creds = await creds_client.request(code)
            assert creds.access_token == "test_access_token"
            assert creds.refresh_token == "test_refresh_token"

    async def test_refresh(self, auth_config: AuthConfig) -> None:
        creds = Credentials(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
        )

        async with ClientSession() as client:
            creds_client = CredentialsClient(
                client=client,
                url=auth_config.token_url,
                client_id=auth_config.client_id,
            )
            new_creds = await creds_client.refresh(creds)
            assert new_creds.access_token == "test_access_token_refreshed"
            assert new_creds.refresh_token == "test_refresh_token"

    async def test_forbidden(
        self, auth_handler: _TestAuthHandler, auth_config: AuthConfig
    ) -> None:
        code = Code(CodeChallenge.create(), "test_code", auth_config.callback_urls[0])

        auth_handler.raise_on_token = HTTPForbidden()

        async with ClientSession() as client:
            creds_client = CredentialsClient(
                client=client,
                url=auth_config.token_url,
                client_id=auth_config.client_id,
            )
            with pytest.raises(AuthException, match="failed to get an access token."):
                await creds_client.request(code)

            creds = Credentials(
                access_token="test_token",
                refresh_token="test_refresh_token",
            )
            with pytest.raises(AuthException, match="failed to get an access token."):
                await creds_client.refresh(creds)


class TestAuthCodeApp:
    @pytest.fixture()
    async def client(self) -> AsyncIterator[ClientSession]:
        async with ClientSession() as client:
            yield client

    async def assert_code_callback_success(
        self,
        future: "asyncio.Future[str]",
        client: ClientSession,
        url: URL,
        redirect_url: Optional[URL] = None,
    ) -> None:
        async with client.get(
            url, params={"code": "testcode"}, allow_redirects=False
        ) as resp:
            if redirect_url:
                assert resp.status == HTTPFound.status_code
                assert resp.headers["Location"] == str(redirect_url)
            else:
                assert resp.status == HTTPOk.status_code
                text = await resp.text()
                assert text == "OK"

        assert await future == "testcode"

    async def test_create_app_server_once(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, success_redirect_url=URL())
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port) as url:
            assert url == URL(f"http://127.0.0.1:{port}")
            await self.assert_code_callback_success(future, client, url)

    async def test_create_app_server_redirect(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        redirect_url = URL("http://redirect.url")
        config = replace(auth_config, success_redirect_url=redirect_url)
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port) as url:
            assert url == URL(f"http://127.0.0.1:{port}")
            await self.assert_code_callback_success(
                future, client, url, redirect_url=redirect_url
            )

    async def test_create_app_server_once_failure(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, success_redirect_url=URL())
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port) as url:
            assert url == URL(f"http://127.0.0.1:{port}")

            async with client.get(url) as resp:
                assert resp.status == HTTPBadRequest.status_code
                text = await resp.text()
                assert text == "The 'code' query parameter is missing."

            with pytest.raises(asyncio.CancelledError):
                await future

    async def test_error_redirect_logout_url(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, failure_redirect_url=URL())
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port) as url:
            assert url == URL(f"http://127.0.0.1:{port}")

            async with client.get(
                url,
                params={"error": "other", "error_description": "Test Other"},
                allow_redirects=False,
            ) as resp:
                assert resp.status == HTTPFound.status_code, await resp.text()
                assert resp.headers["Location"] == str(
                    config.logout_url % {"client_id": config.client_id}
                )

            with pytest.raises(AuthException, match="Test Other"):
                await future

    async def test_error_redirect_failure_url(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = auth_config
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port) as url:
            assert url == URL(f"http://127.0.0.1:{port}")

            async with client.get(
                url,
                params={"error": "other", "error_description": "Test Other"},
                allow_redirects=False,
            ) as resp:
                assert resp.status == HTTPFound.status_code, await resp.text()
                assert resp.headers["Location"] == str(
                    config.logout_url
                    % {
                        "client_id": config.client_id,
                        "returnTo": str(
                            config.failure_redirect_url
                            % {"error": "other", "error_description": "Test Other"}
                        ),
                    }
                )

            with pytest.raises(AuthException, match="Test Other"):
                await future

    async def test_create_app_server(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, success_redirect_url=URL())
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server(app, host="127.0.0.1", ports=[port]) as url:
            assert url == URL(f"http://127.0.0.1:{port}")
            await self.assert_code_callback_success(future, client, url)

    async def test_create_app_server_no_ports(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, success_redirect_url=URL())
        app = create_auth_code_app(future, config)

        port = unused_port()
        async with create_app_server_once(app, host="127.0.0.1", port=port):
            with pytest.raises(RuntimeError, match="No free ports."):
                async with create_app_server(app, ports=[port]):
                    pass

    async def test_create_app_server_port_conflict(
        self, client: ClientSession, auth_config: AuthConfig
    ) -> None:
        future: "asyncio.Future[str]" = asyncio.Future()
        config = replace(auth_config, success_redirect_url=URL())
        app = create_auth_code_app(future, config)

        outer_port = unused_port()
        inner_port = unused_port()
        async with create_app_server(app, ports=[outer_port, inner_port]) as url:
            assert url == URL(f"http://127.0.0.1:{outer_port}")
            async with create_app_server(app, ports=[outer_port, inner_port]) as url:
                assert url == URL(f"http://127.0.0.1:{inner_port}")
                await self.assert_code_callback_success(future, client, url)


class TestWebBrowserCodeClient:
    @pytest.fixture()
    async def client(self) -> AsyncIterator[ClientSession]:
        async with ClientSession() as client:
            yield client

    async def test_request_error(self, auth_config: AuthConfig) -> None:
        config = auth_config

        def _callback(_: URL) -> None:
            async def _acallback() -> None:
                url = config.callback_urls[0].with_query(
                    error="other", error_description="Test Error"
                )
                async with ClientSession() as client:
                    async with client.get(url, allow_redirects=False) as resp:
                        assert resp.status == HTTPFound.status_code, await resp.text()
                        assert resp.headers["Location"] == str(
                            config.logout_url
                            % {
                                "client_id": config.client_id,
                                "returnTo": str(
                                    config.failure_redirect_url
                                    % {
                                        "error": "other",
                                        "error_description": "Test Error",
                                    }
                                ),
                            }
                        )

            asyncio_run_in_thread(_acallback())

        client = WebBrowserCodeClient(config=config, callback=_callback)
        with pytest.raises(AuthException, match="Test Error"):
            await client.request()

    async def test_request_no_code(self, auth_config: AuthConfig) -> None:
        config = replace(auth_config, logout_url=URL())

        def _callback(_: URL) -> None:
            async def _acallback() -> None:
                url = config.callback_urls[0]
                async with ClientSession() as client:
                    async with client.get(url) as resp:
                        assert (
                            resp.status == HTTPBadRequest.status_code
                        ), await resp.text()
                        assert (
                            await resp.text()
                            == "The 'code' query parameter is missing."
                        )

            asyncio_run_in_thread(_acallback())

        client = WebBrowserCodeClient(config=config, callback=_callback)
        with pytest.raises(AuthException, match="failed to get an authorization code"):
            await client.request()

    async def test_request_timeout(self, auth_config: AuthConfig) -> None:
        def _callback(_: URL) -> None:
            pass

        client = WebBrowserCodeClient(
            config=auth_config, callback=_callback, timeout_s=0.0
        )
        with pytest.raises(AuthException, match="failed to get an authorization code"):
            await client.request()

    async def test_request(self, auth_config: AuthConfig) -> None:
        def _callback(_: URL) -> None:
            async def _acallback() -> None:
                url = auth_config.callback_urls[0].with_query(code="test_code")
                async with ClientSession() as client:
                    async with client.get(url, allow_redirects=False) as resp:
                        assert resp.status == HTTPFound.status_code, await resp.text()
                        assert resp.headers["Location"] == str(
                            auth_config.success_redirect_url
                        )

            asyncio_run_in_thread(_acallback())

        client = WebBrowserCodeClient(config=auth_config, callback=_callback)
        code = await client.request()
        assert code.value == "test_code"
        assert code.callback_url == auth_config.callback_urls[0]


class TestHeadlessCodeClient:
    async def test_request_empty_code(self, auth_config: AuthConfig) -> None:
        def _callback(_: str) -> str:
            return ""

        client = HeadlessCodeClient(config=auth_config, callback=_callback)
        with pytest.raises(AuthException, match="No code was provided"):
            await client.request()

    @pytest.mark.parametrize("exception", [EOFError(""), KeyboardInterrupt("")])
    async def test_request_exception(
        self, auth_config: AuthConfig, exception: Exception
    ) -> None:
        def _callback(_: str) -> str:
            raise exception

        client = HeadlessCodeClient(config=auth_config, callback=_callback)
        with pytest.raises(AuthException, match="No code was provided"):
            await client.request()

    async def test_request(self, auth_config: AuthConfig) -> None:
        def _callback(_: str) -> str:
            return "test_code"

        client = HeadlessCodeClient(config=auth_config, callback=_callback)
        code = await client.request()
        assert code.value == "test_code"
        assert code.callback_url == auth_config.headless_callback_url
