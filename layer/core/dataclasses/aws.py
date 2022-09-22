from dataclasses import dataclass


@dataclass(frozen=True)
class AWSCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str


@dataclass(frozen=True)
class S3Path:
    bucket: str
    key: str
