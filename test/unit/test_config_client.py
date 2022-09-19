from typing import AsyncIterator

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import unused_port
from aiohttp.web import Application, Request, Response, json_response
from yarl import URL

from layer.auth import create_app_server_once
from layer.config import ConfigClient
from layer.config.config import AuthConfig, ClientConfig, Config, S3Config


pytestmark = pytest.mark.asyncio


class _TestHandler:
    def __init__(self) -> None:
        self.handle_details_json_response = {
            "auth": {
                "clients": {
                    "sdk": {
                        "callback_urls": [
                            "http://localhost:4123",
                            "http://localhost:6820",
                            "http://localhost:4603",
                            "http://127.0.0.1:4123",
                            "http://127.0.0.1:6820",
                            "http://127.0.0.1:4603",
                        ],
                        "client_id": "nv7k9jFnEHhMVQT6gWRoHMYkEtGUedtz",
                    },
                },
                "domain": "layer-development-rl1abku2vv.eu.auth0.com",
                "rayGatewayDomain": "ray.development.layer.co",
            }
        }

    async def handle_details_json(self, _: Request) -> Response:
        return json_response(self.handle_details_json_response)


@pytest.fixture()
def handler() -> _TestHandler:
    return _TestHandler()


@pytest.fixture()
async def server(handler: _TestHandler) -> AsyncIterator[URL]:
    app = Application()
    app.router.add_get("/__details.json", handler.handle_details_json)
    async with create_app_server_once(app, host="0.0.0.0", port=unused_port()) as url:
        yield url


@pytest.fixture()
async def client(server: URL) -> AsyncIterator[ConfigClient]:
    async with ClientSession() as client:
        yield ConfigClient(url=server, client=client)


class TestConfigClient:
    async def test_get_config(self, server: URL, client: ConfigClient) -> None:
        config = await client.get_config()
        assert config == Config(
            url=server,
            auth=AuthConfig(
                auth_url=URL(server / "oauth" / "authorize"),
                token_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/oauth/token"
                ),
                logout_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/v2/logout"
                ),
                client_id="nv7k9jFnEHhMVQT6gWRoHMYkEtGUedtz",
                audience=str(server),
                headless_callback_url=server / "oauth" / "code",
                callback_urls=[
                    URL("http://127.0.0.1:4123"),
                    URL("http://127.0.0.1:6820"),
                    URL("http://127.0.0.1:4603"),
                ],
                success_redirect_url=server,
                failure_redirect_url=server / "oauth" / "code",
            ),
            client=ClientConfig(
                grpc_gateway_address=f"grpc.{server.host}:{server.port}",
                ray_gateway_address="ray.development.layer.co",
            ),
        )

    async def test_get_config_client_s3(
        self, server: URL, client: ConfigClient, handler: _TestHandler
    ) -> None:
        handler.handle_details_json_response["client"] = {
            "s3_endpoint_url": "http://localhost:12345"
        }
        config = await client.get_config()
        assert config == Config(
            url=server,
            auth=AuthConfig(
                auth_url=URL(server / "oauth" / "authorize"),
                token_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/oauth/token"
                ),
                logout_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/v2/logout"
                ),
                client_id="nv7k9jFnEHhMVQT6gWRoHMYkEtGUedtz",
                audience=str(server),
                headless_callback_url=server / "oauth" / "code",
                callback_urls=[
                    URL("http://127.0.0.1:4123"),
                    URL("http://127.0.0.1:6820"),
                    URL("http://127.0.0.1:4603"),
                ],
                success_redirect_url=server,
                failure_redirect_url=server / "oauth" / "code",
            ),
            client=ClientConfig(
                grpc_gateway_address=f"grpc.{server.host}:{server.port}",
                ray_gateway_address="ray.development.layer.co",
                s3=S3Config(endpoint_url=URL("http://localhost:12345")),
            ),
        )

    async def test_get_config_client_grpc(
        self, server: URL, client: ConfigClient, handler: _TestHandler
    ) -> None:
        handler.handle_details_json_response["client"] = {
            "grpc_gateway_url": "https://localhost:65443",
            "grpc_do_verify_ssl": False,
        }
        config = await client.get_config()
        assert config == Config(
            url=server,
            auth=AuthConfig(
                auth_url=URL(server / "oauth" / "authorize"),
                token_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/oauth/token"
                ),
                logout_url=URL(
                    "https://layer-development-rl1abku2vv.eu.auth0.com/v2/logout"
                ),
                client_id="nv7k9jFnEHhMVQT6gWRoHMYkEtGUedtz",
                audience=str(server),
                headless_callback_url=server / "oauth" / "code",
                callback_urls=[
                    URL("http://127.0.0.1:4123"),
                    URL("http://127.0.0.1:6820"),
                    URL("http://127.0.0.1:4603"),
                ],
                success_redirect_url=server,
                failure_redirect_url=server / "oauth" / "code",
            ),
            client=ClientConfig(
                grpc_gateway_address="localhost:65443",
                ray_gateway_address="ray.development.layer.co",
                grpc_do_verify_ssl=False,
                s3=S3Config(),
            ),
        )

    async def test_get_config_auth_disabled(
        self, server: URL, client: ConfigClient, handler: _TestHandler
    ) -> None:
        del handler.handle_details_json_response["auth"]
        handler.handle_details_json_response["client"] = {
            "grpc_gateway_url": "https://localhost:65443",
            "grpc_do_verify_ssl": False,
        }
        config = await client.get_config()
        assert config == Config(
            url=server,
            auth=AuthConfig.create_disabled(),
            client=ClientConfig(
                grpc_gateway_address="localhost:65443",
                ray_gateway_address=f"ray.{server.host}:{server.port}",
                grpc_do_verify_ssl=False,
                s3=S3Config(),
            ),
        )

    def test_trailing_slash_in_url_is_removed_from_auth_config_audience(
        self, client: ConfigClient
    ) -> None:
        payload = {
            "clients": {
                "sdk": {
                    "callback_urls": [],
                    "client_id": "abc",
                },
            },
            "domain": "xyz.eu.auth0.com",
        }
        result = client._create_auth_config(  # pylint: disable=protected-access
            url=URL("https://foo.layer.co/"), payload=payload
        )
        assert "https://foo.layer.co" == result.audience
