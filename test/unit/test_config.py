import time
from pathlib import Path
from typing import Any, Dict

import jwt
import pytest
from yarl import URL

from layer.config import (
    AuthConfig,
    ClientConfig,
    Config,
    ConfigRecord,
    ConfigStore,
    Credentials,
    S3Config,
)
from layer.exceptions.exceptions import (
    InvalidConfigurationError,
    MissingConfigurationError,
)


class TestCredentials:
    def test_is_access_token_expired_no_exp_is_not_expired(self) -> None:
        access_token = jwt.encode({}, "secret", algorithm="HS256")
        credentials = Credentials(access_token=access_token, refresh_token="")
        assert not credentials.is_access_token_expired

    def test_is_access_token_expired_exp_now_is_expired(self) -> None:
        expiration_time = time.time()
        access_token = jwt.encode({"exp": expiration_time}, "secret", algorithm="HS256")
        credentials = Credentials(access_token=access_token, refresh_token="")
        assert credentials.is_access_token_expired

    def test_is_access_token_expired_exp_in_30_min_is_expired(self) -> None:
        expiration_time = time.time() + 30 * 60
        access_token = jwt.encode({"exp": expiration_time}, "secret", algorithm="HS256")
        credentials = Credentials(access_token=access_token, refresh_token="")
        assert credentials.is_access_token_expired

    def test_is_access_token_expired_exp_in_90_min_is_not_expired(self) -> None:
        expiration_time = time.time() + 90 * 60
        access_token = jwt.encode({"exp": expiration_time}, "secret", algorithm="HS256")
        credentials = Credentials(access_token=access_token, refresh_token="")
        assert not credentials.is_access_token_expired


class TestConfigStore:
    @pytest.fixture()
    def config(self) -> Config:
        return Config(
            url=URL("https://development.layer.co"),
            auth=AuthConfig(
                auth_url=URL("https://tenant.auth0.com/oauth/authorize"),
                token_url=URL("https://tenant.auth0.com/oauth/token"),
                logout_url=URL("https://tenant.auth0.com/v2/logout"),
                client_id="testclientid",
                audience="https://development.layer.co",
                headless_callback_url=URL("https://development.layer.co/oauth/code"),
                callback_urls=[
                    URL("http://127.0.0.1:1234"),
                    URL("http://127.0.0.1:1234"),
                ],
                success_redirect_url=URL("https://development.layer.co"),
                failure_redirect_url=URL("https://development.layer.co/oauth/code"),
            ),
            credentials=Credentials(
                access_token="testaccesstoken",
                refresh_token="testrefreshtoken",
            ),
            client=ClientConfig(
                grpc_gateway_address="grpcgatewayaddress",
                ray_gateway_address="raygatewayaddress",
                access_token="testaccesstoken",
            ),
        )

    def test_save_load(self, tmp_path: Path, config: Config) -> None:
        path = tmp_path / "dir" / "config.json"

        store = ConfigStore(path)
        store.save(config)
        new_config = store.load()
        assert new_config == config

    def test_load_malformed(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidConfigurationError):
            path = tmp_path / "dir" / "config.json"
            path.parent.mkdir(parents=True)
            with open(path, "w") as f:
                f.write("malformed")
            ConfigStore(path).load()

    def test_delete(self, tmp_path: Path, config: Config) -> None:
        with pytest.raises(MissingConfigurationError):
            path = tmp_path / "dir" / "config.json"

            store = ConfigStore(path)
            store.save(config)
            store.delete()
            store.load()

    def test_delete_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "dir" / "config.json"

        store = ConfigStore(path)
        store.delete()


class TestConfigRecord:
    def test_from_auth_disabled(self) -> None:
        record = ConfigRecord.from_auth(AuthConfig.create_disabled())
        assert record == {}

    def test_to_auth_disabled(self) -> None:
        config = ConfigRecord.to_auth({})
        assert config == AuthConfig.create_disabled()

    def test_to_auth_no_logout_url(self) -> None:
        base_url = URL("https://auth.development.layer.co")
        url = URL("https://development.layer.co")
        record: Dict[str, Any] = {
            "auth_url": str(base_url / "oauth" / "authorize"),
            "token_url": str(base_url / "oauth" / "token"),
            "client_id": "test_client_id",
            "audience": str(url),
            "headless_callback_url": str(url / "oauth" / "code"),
            "callback_urls": ["http://127.0.0.1:4123"],
            "success_redirect_url": str(url),
        }
        assert ConfigRecord.to_auth(record) == AuthConfig(
            auth_url=URL(record["auth_url"]),
            token_url=URL(record["token_url"]),
            logout_url=base_url / "v2" / "logout",
            client_id=record["client_id"],
            audience=record["audience"],
            headless_callback_url=URL(record["headless_callback_url"]),
            callback_urls=[URL(url) for url in record["callback_urls"]],
            success_redirect_url=URL(record["success_redirect_url"]),
            failure_redirect_url=URL(record["headless_callback_url"]),
        )

    def test_to_auth(self) -> None:
        base_url = URL("https://auth.development.layer.co")
        url = URL("https://development.layer.co")
        record: Dict[str, Any] = {
            "auth_url": str(base_url / "oauth" / "authorize"),
            "token_url": str(base_url / "oauth" / "token"),
            "logout_url": str(base_url / "v2" / "whatever"),
            "client_id": "test_client_id",
            "audience": str(url),
            "headless_callback_url": str(url / "oauth" / "code"),
            "callback_urls": ["http://127.0.0.1:4123"],
            "success_redirect_url": str(url),
            "failure_redirect_url": str(url / "oauth" / "nope"),
        }
        assert ConfigRecord.to_auth(record) == AuthConfig(
            auth_url=URL(record["auth_url"]),
            token_url=URL(record["token_url"]),
            logout_url=URL(base_url / "v2" / "whatever"),
            client_id=record["client_id"],
            audience=record["audience"],
            headless_callback_url=URL(record["headless_callback_url"]),
            callback_urls=[URL(url) for url in record["callback_urls"]],
            success_redirect_url=URL(record["success_redirect_url"]),
            failure_redirect_url=url / "oauth" / "nope",
        )

    def test_to_client(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.to_client(
            {"grpc_gateway_address": address, "ray_gateway_address": address},
            "testaccesstoken",
        ) == ClientConfig(
            grpc_gateway_address=address,
            ray_gateway_address=address,
            access_token="testaccesstoken",
            s3=S3Config(),
        )

    def test_to_client_s3(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.to_client(
            {
                "grpc_gateway_address": address,
                "ray_gateway_address": address,
                "s3_endpoint_url": "http://localhost:12345",
            },
            "testaccesstoken",
        ) == ClientConfig(
            grpc_gateway_address=address,
            ray_gateway_address=address,
            access_token="testaccesstoken",
            s3=S3Config(endpoint_url=URL("http://localhost:12345")),
        )

    def test_to_client_grpc_do_verify_ssl(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.to_client(
            {
                "grpc_gateway_address": address,
                "ray_gateway_address": address,
                "grpc_do_verify_ssl": False,
            },
            "testaccesstoken",
        ) == ClientConfig(
            grpc_gateway_address=address,
            ray_gateway_address=address,
            grpc_do_verify_ssl=False,
            access_token="testaccesstoken",
            s3=S3Config(),
        )

    def test_from_client(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.from_client(
            ClientConfig(
                grpc_gateway_address=address,
                ray_gateway_address=address,
                access_token="testaccesstoken",
                s3=S3Config(),
            )
        ) == {"grpc_gateway_address": address, "ray_gateway_address": address}

    def test_from_client_s3(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.from_client(
            ClientConfig(
                grpc_gateway_address=address,
                ray_gateway_address=address,
                access_token="testaccesstoken",
                s3=S3Config(endpoint_url=URL("http://localhost:12345")),
            )
        ) == {
            "grpc_gateway_address": address,
            "ray_gateway_address": address,
            "s3_endpoint_url": "http://localhost:12345",
        }

    def test_from_client_grpc_do_verify_ssl(self) -> None:
        address = "localhost:54321"
        assert ConfigRecord.from_client(
            ClientConfig(
                grpc_gateway_address=address,
                ray_gateway_address=address,
                grpc_do_verify_ssl=False,
                access_token="testaccesstoken",
                s3=S3Config(),
            )
        ) == {
            "grpc_gateway_address": address,
            "ray_gateway_address": address,
            "grpc_do_verify_ssl": False,
        }

    def test_from_credentials_empty(self) -> None:
        record = ConfigRecord.from_credentials(Credentials.create_empty())
        assert record == {"access_token": "", "refresh_token": ""}

    def test_to_credentials_empty(self) -> None:
        credentials = ConfigRecord.to_credentials({})
        assert credentials == Credentials.create_empty()
