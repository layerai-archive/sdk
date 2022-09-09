from layerapi.api.value.aws_credentials_pb2 import AwsCredentials as PBAwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path as PBS3Path

from layer.contracts.aws import AWSCredentials, S3Path


def from_aws_credentials(credentials: PBAwsCredentials) -> AWSCredentials:
    return AWSCredentials(
        access_key_id=credentials.access_key_id,
        secret_access_key=credentials.secret_access_key,
        session_token=credentials.session_token,
    )


def to_aws_credentials(credentials: AWSCredentials) -> PBAwsCredentials:
    return PBAwsCredentials(
        access_key_id=credentials.access_key_id,
        secret_access_key=credentials.secret_access_key,
        session_token=credentials.session_token,
    )


def from_s3_path(s3_path: PBS3Path) -> S3Path:
    return S3Path(
        bucket=s3_path.bucket,
        key=s3_path.key,
    )


def to_s3_path(s3_path: S3Path) -> PBS3Path:
    return PBS3Path(
        bucket=s3_path.bucket,
        key=s3_path.key,
    )
