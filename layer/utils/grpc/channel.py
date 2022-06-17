import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import grpc

from .interceptors import (
    GRPCErrorClientInterceptor,
    LogRpcCallsInterceptor,
    RequestIdInterceptor,
)


def create_grpc_channel(
    address: str,
    access_token: str,
    *,
    do_verify_ssl: bool = True,
    logs_file_path: Path,
    options: Optional[List[Tuple[str, Union[str, int]]]] = None,
) -> Any:
    # https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
    # https://github.com/grpc/proposal/blob/master/A8-client-side-keepalive.md
    if options is None:
        options = []
    options = options.copy()
    ssl_config = create_grpc_ssl_config(address, do_verify_ssl=do_verify_ssl)
    if ssl_config.hostname_override:
        options.append(("grpc.ssl_target_name_override", ssl_config.hostname_override))
    json_config = json.dumps(
        {
            "methodConfig": [
                {
                    "name": [{}],
                    "retryPolicy": {
                        "maxAttempts": 5,
                        "initialBackoff": "0.1s",
                        "maxBackoff": "5s",
                        "backoffMultiplier": 2,
                        "retryableStatusCodes": ["UNAVAILABLE"],
                    },
                }
            ]
        }
    )
    options.append(("grpc.enable_retries", 1))
    options.append(("grpc.service_config", json_config))
    options.append(("grpc.keepalive_time_ms", 60000))
    options.append(("grpc.keepalive_timeout_ms", 5000))
    options.append(("grpc.keepalive_permit_without_calls", 1))
    options.append(("grpc.http2.max_pings_without_data", 0))
    credentials = grpc.ssl_channel_credentials(ssl_config.cadata)

    client_interceptors = [
        RequestIdInterceptor(),
        GRPCErrorClientInterceptor(),
        LogRpcCallsInterceptor(logs_file_path),
    ]

    return grpc.intercept_channel(
        grpc.secure_channel(
            address,
            grpc.composite_channel_credentials(
                credentials,
                grpc.access_token_call_credentials(access_token),
            ),
            options,
        ),
        *client_interceptors,
    )


@dataclass(frozen=True)
class GRPCSSLConfig:
    cadata: Optional[bytes] = None
    hostname_override: Optional[str] = None


def _load_default_ssl_certs() -> bytes:
    ssl_context = ssl.create_default_context()
    der_certs = ssl_context.get_ca_certs(binary_form=True)  # certs are in der format
    pem_certs = [ssl.DER_cert_to_PEM_cert(der_cert) for der_cert in der_certs]  # type: ignore
    return "\n".join(pem_certs).encode()


def create_grpc_ssl_config(
    address: str, *, do_verify_ssl: bool = True, do_force_cadata_load: bool = False
) -> GRPCSSLConfig:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    cadata: Optional[bytes] = None

    if do_verify_ssl:
        if do_force_cadata_load:
            cadata = _load_default_ssl_certs()
        return GRPCSSLConfig(cadata=cadata)

    host, port = address.split(":", 1)
    cadata = ssl.get_server_certificate((host, int(port))).encode()
    cert = x509.load_pem_x509_certificate(cadata)
    hostname_override = cert.extensions.get_extension_for_oid(
        ExtensionOID.SUBJECT_ALTERNATIVE_NAME
    ).value.get_values_for_type(x509.DNSName)[0]
    return GRPCSSLConfig(cadata=cadata, hostname_override=hostname_override)
