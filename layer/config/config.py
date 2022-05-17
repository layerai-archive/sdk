import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import jwt
from yarl import URL

from layer.exceptions.exceptions import (
    ConfigError,
    InvalidConfigurationError,
    MissingConfigurationError,
)
from layer.run import UserSessionId


DEFAULT_LAYER_PATH = Path(os.getenv("LAYER_DEFAULT_PATH", Path.home() / ".layer"))
DEFAULT_PATH = DEFAULT_LAYER_PATH / "config.json"
DEFAULT_LOGS_DIR = DEFAULT_LAYER_PATH / "logs"
DEFAULT_FUNC_PATH = DEFAULT_LAYER_PATH / "functions"
DEFAULT_URL = URL("https://app.layer.ai")
ROOT_LAYER_URL = URL("https://layer.co")


@dataclass(frozen=True)
class ServiceConfig:
    address: str = ""

    @property
    def is_enabled(self) -> bool:
        return bool(self.address)


@dataclass(frozen=True)
class ProjectServiceConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class DataCatalogConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class ModelCatalogConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class ModelTrainingConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class AccountServiceConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class FlowManagerServiceConfig(ServiceConfig):
    pass


@dataclass(frozen=True)
class UserLogsServiceConfig(ServiceConfig):
    @property
    def max_receive_message_length(self) -> int:
        return 100 * 1024 * 1024


@dataclass(frozen=True)
class S3Config:
    endpoint_url: Optional[URL] = None

    @classmethod
    def create_default(cls) -> "S3Config":
        return cls()


def _default_logs_file_path() -> Path:
    local_now = datetime.now().strftime("%Y%m%dT%H%M%S")
    logs_file_path = Path.joinpath(
        DEFAULT_LOGS_DIR, f"{local_now}-session-{UserSessionId()}.log"
    )
    return logs_file_path


@dataclass(frozen=True)
class LogsConfig:
    logs_file_path: Path = _default_logs_file_path()


@dataclass(frozen=True)
class ClientConfig:
    data_catalog: DataCatalogConfig
    model_catalog: ModelCatalogConfig
    model_training: ModelTrainingConfig
    account_service: AccountServiceConfig
    flow_manager: FlowManagerServiceConfig
    user_logs: UserLogsServiceConfig
    project_service: ProjectServiceConfig
    grpc_gateway_address: str = ""
    access_token: str = ""
    grpc_do_verify_ssl: bool = True
    logs_file_path: Path = _default_logs_file_path()
    s3: S3Config = S3Config.create_default()

    def with_access_token(self, access_token: str) -> "ClientConfig":
        return replace(self, access_token=access_token)


@dataclass(frozen=True)
class AuthConfig:
    auth_url: URL
    token_url: URL
    logout_url: URL

    client_id: str
    audience: str

    headless_callback_url: URL

    callback_urls: Sequence[URL]
    success_redirect_url: URL
    failure_redirect_url: URL

    @classmethod
    def create_disabled(cls) -> "AuthConfig":
        return AuthConfig(
            auth_url=URL(),
            token_url=URL(),
            logout_url=URL(),
            client_id="",
            audience="",
            headless_callback_url=URL(),
            callback_urls=[],
            success_redirect_url=URL(),
            failure_redirect_url=URL(),
        )

    @property
    def is_enabled(self) -> bool:
        return bool(self.auth_url)

    @property
    def callback_host(self) -> str:
        assert self.callback_urls[0].host
        return self.callback_urls[0].host

    @property
    def callback_ports(self) -> List[int]:
        return [url.port for url in self.callback_urls if url.port]


@dataclass(frozen=True)
class Credentials:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)

    # This is used to assume the tokens expire a bit earlier than their actual
    # expiration time. Therefore we will never send an expired token.
    _expiration_margin: ClassVar[float] = 60.0 * 60.0  # 1 hour

    @classmethod
    def create_empty(cls) -> "Credentials":
        return cls(access_token="", refresh_token="")

    @property
    def _access_token_expiration_time(self) -> float:
        return (
            float(jwt.decode(self.access_token, verify=False).get("exp", float("inf")))
            - self._expiration_margin
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self.access_token)

    @property
    def is_access_token_expired(self) -> bool:
        return self.is_empty or time.time() >= self._access_token_expiration_time

    @property
    def is_authenticated_outside_organization(self) -> bool:
        # todo: update this to account_id along with LAY-2716
        return self.is_empty or f"{ROOT_LAYER_URL}/organization_id" not in jwt.decode(
            self.access_token, verify=False
        )


def get_config(name: str, record: Dict[str, Any]) -> Any:
    if name not in record:
        raise ConfigError(
            "Missing configuration parameter. Make sure you have the latest release and login again."
        )
    return record[name]


def get_config_or_default(name: str, default_val: Any, record: Dict[str, Any]) -> Any:
    if name not in record:
        return default_val
    return record[name]


@dataclass(frozen=True)
class Config:
    url: URL
    client: ClientConfig
    auth: AuthConfig
    credentials: Credentials = Credentials.create_empty()
    is_guest: bool = False

    def with_credentials(self, creds: Credentials) -> "Config":
        return replace(
            self,
            credentials=creds,
            client=replace(self.client, access_token=creds.access_token),
        )


class ConfigRecord:
    @classmethod
    def from_auth(cls, config: AuthConfig) -> Dict[str, Any]:
        if not config.is_enabled:
            return {}
        return {
            "auth_url": str(config.auth_url),
            "token_url": str(config.token_url),
            "logout_url": str(config.logout_url),
            "client_id": config.client_id,
            "audience": config.audience,
            "headless_callback_url": str(config.headless_callback_url),
            "callback_urls": [str(url) for url in config.callback_urls],
            "success_redirect_url": str(config.success_redirect_url),
            "failure_redirect_url": str(config.failure_redirect_url),
        }

    @classmethod
    def to_auth(cls, record: Dict[str, Any]) -> AuthConfig:
        if not record:
            return AuthConfig.create_disabled()
        auth_url = URL(record["auth_url"])
        logout_url = auth_url.with_path("/v2/logout")
        if record.get("logout_url"):
            logout_url = URL(record["logout_url"])
        headless_callback_url = URL(record["headless_callback_url"])
        failure_redirect_url = headless_callback_url
        if record.get("failure_redirect_url"):
            failure_redirect_url = URL(record["failure_redirect_url"])
        return AuthConfig(
            auth_url=URL(record["auth_url"]),
            token_url=URL(record["token_url"]),
            logout_url=logout_url,
            client_id=record["client_id"],
            audience=record["audience"],
            headless_callback_url=headless_callback_url,
            callback_urls=[URL(url) for url in record["callback_urls"]],
            success_redirect_url=URL(record["success_redirect_url"]),
            failure_redirect_url=failure_redirect_url,
        )

    @classmethod
    def from_credentials(cls, creds: Credentials) -> Dict[str, Any]:
        return {
            "access_token": creds.access_token,
            "refresh_token": creds.refresh_token,
        }

    @classmethod
    def to_credentials(cls, record: Dict[str, Any]) -> Credentials:
        if not record:
            return Credentials.create_empty()
        return Credentials(
            access_token=record["access_token"],
            refresh_token=record["refresh_token"],
        )

    @classmethod
    def from_client(cls, config: ClientConfig) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "grpc_gateway_address": config.grpc_gateway_address,
        }
        if not config.grpc_do_verify_ssl:
            record["grpc_do_verify_ssl"] = config.grpc_do_verify_ssl
        if config.s3.endpoint_url:
            record["s3_endpoint_url"] = str(config.s3.endpoint_url)
        return record

    @classmethod
    def to_client(cls, record: Dict[str, Any], access_token: str) -> ClientConfig:
        grpc_gateway_address = get_config("grpc_gateway_address", record)
        grpc_do_verify_ssl = record.get("grpc_do_verify_ssl", True)
        if "s3_endpoint_url" in record:
            s3_config = S3Config(endpoint_url=URL(record["s3_endpoint_url"]))
        else:
            s3_config = S3Config.create_default()
        return ClientConfig(
            data_catalog=DataCatalogConfig(address=grpc_gateway_address),
            model_catalog=ModelCatalogConfig(address=grpc_gateway_address),
            model_training=ModelTrainingConfig(address=grpc_gateway_address),
            account_service=AccountServiceConfig(address=grpc_gateway_address),
            flow_manager=FlowManagerServiceConfig(address=grpc_gateway_address),
            user_logs=UserLogsServiceConfig(address=grpc_gateway_address),
            project_service=ProjectServiceConfig(address=grpc_gateway_address),
            grpc_gateway_address=grpc_gateway_address,
            access_token=access_token,
            grpc_do_verify_ssl=grpc_do_verify_ssl,
            s3=s3_config,
        )

    @classmethod
    def from_config(cls, config: Config) -> Dict[str, Any]:
        assert config.credentials
        return {
            "is_guest": config.is_guest,
            "url": str(config.url),
            "auth": cls.from_auth(config.auth),
            "credentials": cls.from_credentials(config.credentials),
            "client": cls.from_client(config.client),
        }

    @classmethod
    def to_config(cls, record: Dict[str, Any]) -> Config:
        is_guest = bool(get_config_or_default("is_guest", False, record))
        url = URL(get_config("url", record))
        creds = cls.to_credentials(get_config("credentials", record))
        auth = cls.to_auth(get_config("auth", record))
        client = cls.to_client(
            record=get_config("client", record), access_token=creds.access_token
        )
        return Config(
            is_guest=is_guest,
            url=url,
            auth=auth,
            credentials=creds,
            client=client,
        )


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, config: Config) -> None:
        record = ConfigRecord.from_config(config)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(record, f)
        self._path.chmod(0o600)

    def load(self) -> Config:
        try:
            with open(self._path, "r") as f:
                return ConfigRecord.to_config(json.load(f))
        except IOError:
            raise MissingConfigurationError(self._path)
        except Exception:
            raise InvalidConfigurationError(self._path)

    def delete(self) -> None:
        try:
            self._path.unlink()  # same as unlink(self, missing_ok=True), but works on Python < 3.8
        except FileNotFoundError:
            pass
