from typing import TYPE_CHECKING, Any, Dict, Optional

from yarl import URL

from layer.config import (
    AccountServiceConfig,
    AuthConfig,
    ClientConfig,
    Config,
    DataCatalogConfig,
    FlowManagerServiceConfig,
    ModelCatalogConfig,
    ModelTrainingConfig,
    ProjectServiceConfig,
    S3Config,
    UserLogsServiceConfig,
)


if TYPE_CHECKING:
    from aiohttp import ClientSession


class ConfigClient:
    def __init__(
        self, *, url: URL, client: "ClientSession", do_verify_ssl: bool = True
    ) -> None:
        self._url = url
        self._details_url = url / "__details.json"
        self._client = client
        self._do_verify_ssl = do_verify_ssl
        self._client_ssl_param = None if do_verify_ssl else False

    async def get_config(self) -> Config:
        async with self._client.get(
            self._details_url, ssl=self._client_ssl_param
        ) as resp:
            payload = await resp.json()
            return Config(
                url=self._url,
                auth=self._create_auth_config(payload.get("auth", {})),
                client=self._create_client_config(payload.get("client", {})),
            )

    def _create_auth_config(
        self, payload: Dict[str, Any], url: Optional[URL] = None
    ) -> AuthConfig:
        if not payload:
            return AuthConfig.create_disabled()
        if not url:
            url = self._url
        base_url = URL.build(scheme="https", host=payload["domain"])
        sdk_client_payload = payload["clients"]["sdk"]
        callback_urls = [
            URL(callback_url)
            for callback_url in sdk_client_payload["callback_urls"]
            if "127.0.0.1" in callback_url
        ]
        return AuthConfig(
            auth_url=url / "oauth" / "authorize",
            token_url=base_url / "oauth" / "token",
            logout_url=base_url / "v2" / "logout",
            client_id=sdk_client_payload["client_id"],
            # Our 'audience' of Auth0 OAuth config should not have a trailing slash, otherwise it fails.
            audience=str(self._remove_trailing_slash(url)),
            headless_callback_url=url / "oauth" / "code",
            callback_urls=callback_urls,
            success_redirect_url=url,
            failure_redirect_url=url / "oauth" / "code",
        )

    @staticmethod
    def _remove_trailing_slash(url: URL) -> URL:
        return URL(str(url).rstrip("/"))

    def _create_client_config(self, payload: Dict[str, Any]) -> ClientConfig:
        if "grpc_gateway_url" in payload:
            url = URL(payload["grpc_gateway_url"])
            do_verify_ssl = payload.get("grpc_do_verify_ssl", True)
        else:
            url = self._url.with_host(f"grpc.{self._url.host}")
            do_verify_ssl = self._do_verify_ssl
        grpc_gateway_address = f"{url.host}:{url.port}"
        return ClientConfig(
            data_catalog=DataCatalogConfig(address=grpc_gateway_address),
            model_catalog=ModelCatalogConfig(address=grpc_gateway_address),
            model_training=ModelTrainingConfig(address=grpc_gateway_address),
            account_service=AccountServiceConfig(address=grpc_gateway_address),
            flow_manager=FlowManagerServiceConfig(address=grpc_gateway_address),
            user_logs=UserLogsServiceConfig(address=grpc_gateway_address),
            project_service=ProjectServiceConfig(address=grpc_gateway_address),
            grpc_gateway_address=grpc_gateway_address,
            grpc_do_verify_ssl=do_verify_ssl,
            s3=self._create_s3_config(payload),
        )

    def _create_s3_config(self, payload: Dict[str, Any]) -> S3Config:
        s3_endpoint_url = None
        if payload.get("s3_endpoint_url"):
            s3_endpoint_url = URL(payload["s3_endpoint_url"])
        return S3Config(endpoint_url=s3_endpoint_url)
