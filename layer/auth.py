import abc
import asyncio
import base64
import errno
import hashlib
import secrets
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Sequence
from urllib.parse import urlencode

from yarl import URL

from layer.config import AuthConfig, Credentials
from layer.exceptions.exceptions import AuthException


if TYPE_CHECKING:
    from aiohttp import ClientSession
    from aiohttp.web import Application, Request, Response


EMPTY_URL = URL()


def urlsafe_unpadded_b64encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


@asynccontextmanager
async def create_app_server_once(
    app: "Application", *, host: str = "127.0.0.1", port: int = 8080
) -> AsyncIterator[URL]:
    from aiohttp.web import AppRunner, TCPSite

    runner = AppRunner(app, access_log=None)
    try:
        await runner.setup()
        site = TCPSite(runner, host, port, shutdown_timeout=0.0)
        await site.start()
        yield URL(site.name)
    finally:
        await runner.shutdown()
        await runner.cleanup()


@asynccontextmanager
async def create_app_server(
    app: "Application", *, host: str = "127.0.0.1", ports: Sequence[int] = (8080,)
) -> AsyncIterator[URL]:
    for port in ports:
        try:
            async with create_app_server_once(app, host=host, port=port) as url:
                yield url
            return
        except OSError as err:
            if err.errno != errno.EADDRINUSE:
                raise
    raise RuntimeError("No free ports.")


@dataclass(frozen=True)
class CodeChallenge:
    verifier: str
    value: str
    method: ClassVar[str] = "S256"

    @classmethod
    def create(cls) -> "CodeChallenge":
        verifier = urlsafe_unpadded_b64encode(secrets.token_bytes(32))
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = urlsafe_unpadded_b64encode(digest)
        return cls(
            verifier=verifier,
            value=challenge,
        )


@dataclass(frozen=True)
class Code:
    challenge: CodeChallenge
    value: str
    callback_url: URL


class CodeCallbackHandler:
    def __init__(
        self,
        future: "asyncio.Future[str]",
        config: AuthConfig,
    ) -> None:
        self._future = future
        self._config = config

    async def handle(self, request: "Request") -> "Response":
        from aiohttp.web import HTTPBadRequest, HTTPFound, Response

        if "error" in request.query:
            await self._handle_error(request)

        code = request.query.get("code")

        if not code:
            self._future.cancel()
            raise HTTPBadRequest(text="The 'code' query parameter is missing.")

        if not self._future.cancelled():
            self._future.set_result(code)

        if self._config.success_redirect_url:
            raise HTTPFound(self._config.success_redirect_url)
        return Response(text="OK")

    async def _handle_error(self, request: "Request") -> None:
        from aiohttp.web import HTTPFound

        error = request.query["error"]
        description = request.query.get("error_description", "")

        if not self._future.cancelled():
            self._future.set_exception(AuthException(description))

        raise HTTPFound(self._generate_logout_url(error, description))

    def _generate_logout_url(self, error: str, description: str) -> URL:
        query = {"client_id": self._config.client_id}
        if self._config.failure_redirect_url:
            query["returnTo"] = str(
                self._config.failure_redirect_url.with_query(
                    error=error, error_description=description
                )
            )
        return self._config.logout_url.with_query(**query)


def create_auth_code_app(
    future: "asyncio.Future[str]",
    config: AuthConfig,
) -> "Application":
    from aiohttp.web import Application

    app = Application()
    handler = CodeCallbackHandler(future, config)
    app.router.add_get("/", handler.handle)
    return app


class CodeClient(abc.ABC):
    def __init__(self, *, config: AuthConfig) -> None:
        self._config = config

    def _generate_auth_url(
        self, code_challenge: CodeChallenge, callback_url: URL
    ) -> URL:
        return self._config.auth_url.with_query(
            response_type="code",
            code_challenge=code_challenge.value,
            code_challenge_method=code_challenge.method,
            client_id=self._config.client_id,
            redirect_uri=str(callback_url),
            scope="offline_access",
            audience=self._config.audience,
        )

    @abc.abstractmethod
    async def request(self) -> Code:
        pass


class WebBrowserCodeClient(CodeClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._timeout_s = kwargs.pop("timeout_s", 5 * 60.0)
        self._code_callback = kwargs.pop("callback", webbrowser.open_new)
        super().__init__(*args, **kwargs)

    async def request(self) -> Code:
        code_challenge = CodeChallenge.create()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = asyncio.Future()
        app = create_auth_code_app(future, self._config)
        try:
            async with create_app_server(
                app, host=self._config.callback_host, ports=self._config.callback_ports
            ) as callback_url:
                auth_url = self._generate_auth_url(code_challenge, callback_url)
                await loop.run_in_executor(None, self._code_callback, str(auth_url))
                try:
                    await asyncio.wait_for(future, self._timeout_s)
                except (asyncio.TimeoutError, asyncio.CancelledError) as err:
                    raise AuthException("failed to get an authorization code") from err
        except (OSError, RuntimeError) as err:
            raise AuthException("failed to start a code callback server") from err
        return Code(
            challenge=code_challenge, value=future.result(), callback_url=callback_url
        )


class HeadlessCodeClient(CodeClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._code_callback = kwargs.pop("callback", input)
        super().__init__(*args, **kwargs)

    async def request(self) -> Code:
        code_challenge = CodeChallenge.create()
        loop = asyncio.get_event_loop()
        callback_url = self._config.headless_callback_url
        auth_url = self._generate_auth_url(code_challenge, callback_url)
        print(
            "Please open the following link in your web browser. "
            "Once logged in, copy the code and paste it here."
        )
        print(auth_url)
        try:
            value = await loop.run_in_executor(None, self._code_callback, "Code: ")
            if not value:
                raise ValueError("Empty code")
        except (EOFError, KeyboardInterrupt, ValueError) as exc:
            raise AuthException("No code was provided") from exc
        return Code(challenge=code_challenge, value=value, callback_url=callback_url)


class CredentialsClient:
    def __init__(self, *, client: "ClientSession", url: URL, client_id: str) -> None:
        self._client = client

        self._url = url
        self._client_id = client_id

    async def request(self, code: Code) -> Credentials:
        from aiohttp import ClientResponseError

        payload = {
            "grant_type": "authorization_code",
            "code_verifier": code.challenge.verifier,
            "code": code.value,
            "client_id": self._client_id,
            "redirect_uri": str(code.callback_url),
        }
        async with self._client.post(
            self._url,
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
            data=urlencode(payload),
        ) as resp:
            try:
                resp.raise_for_status()
            except ClientResponseError as exc:
                raise AuthException("failed to get an access token.") from exc
            resp_payload = await resp.json()
            return Credentials(
                access_token=resp_payload["access_token"],
                refresh_token=resp_payload["refresh_token"],
            )

    async def refresh(self, creds: Credentials) -> Credentials:
        from aiohttp import ClientResponseError

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": creds.refresh_token,
            "client_id": self._client_id,
        }
        async with self._client.post(
            self._url,
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
            data=urlencode(payload),
        ) as resp:
            try:
                resp.raise_for_status()
            except ClientResponseError as exc:
                raise AuthException("failed to get an access token.") from exc
            resp_payload = await resp.json()
            return Credentials(
                access_token=resp_payload["access_token"],
                refresh_token=creds.refresh_token,
            )
