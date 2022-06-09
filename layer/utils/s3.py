import os
import posixpath
from mimetypes import guess_type
from pathlib import Path
from typing import Optional

from layerapi.api.value.aws_credentials_pb2 import AwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path
from yarl import URL

from layer.contracts.tracker import ResourceTransferState


class S3Util:
    @staticmethod
    def download_dir(
        local_dir: Path,
        s3_path: S3Path,
        credentials: Optional[AwsCredentials] = None,
        *,
        endpoint_url: Optional[URL] = None,
        state: ResourceTransferState,
    ) -> None:
        import boto3

        s3_kwargs = {"endpoint_url": endpoint_url and str(endpoint_url)}

        if credentials:
            s3_kwargs.update(
                {
                    "aws_access_key_id": credentials.access_key_id,
                    "aws_secret_access_key": credentials.secret_access_key,
                    "aws_session_token": credentials.session_token,
                }
            )
        s3 = boto3.resource("s3", **s3_kwargs)  # type: ignore
        bucket = s3.Bucket(s3_path.bucket)

        to_download = []
        total_num_files = 0
        total_file_bytes = 0
        for obj in bucket.objects.filter(Prefix=s3_path.key):
            target = os.path.join(local_dir, os.path.relpath(obj.key, s3_path.key))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == "/":
                continue

            total_num_files += 1
            total_file_bytes += obj.size
            to_download.append(
                {
                    "args": [obj.key, target],
                    "kwargs": {
                        "Callback": state.increment_transferred_resource_size_bytes
                    },
                }
            )

        state.total_num_files = total_num_files
        state.total_resource_size_bytes = total_file_bytes
        for function_params in to_download:
            bucket.download_file(*function_params["args"], **function_params["kwargs"])

    @staticmethod
    def upload_dir(
        local_dir: Path,
        credentials: AwsCredentials,
        s3_path: S3Path,
        *,
        endpoint_url: Optional[URL] = None,
        state: Optional[ResourceTransferState] = None,
    ) -> None:
        import boto3
        from mlflow.utils.file_utils import relative_path_to_artifact_path

        if not state:
            state = ResourceTransferState()

        s3_kwargs = {"endpoint_url": endpoint_url and str(endpoint_url)}

        if credentials:
            s3_kwargs.update(
                {
                    "aws_access_key_id": credentials.access_key_id,
                    "aws_secret_access_key": credentials.secret_access_key,
                    "aws_session_token": credentials.session_token,
                }
            )

        s3_client = boto3.client("s3", **s3_kwargs)  # type: ignore
        dest_path = s3_path.key
        absolute_path = os.path.abspath(local_dir)
        total_file_bytes = 0
        total_num_files = 0
        to_upload = []
        for (root, _, filenames) in os.walk(absolute_path):
            upload_path = dest_path
            if root != absolute_path:
                rel_path = os.path.relpath(root, absolute_path)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                filename = os.path.join(root, f)
                total_num_files += 1
                total_file_bytes += os.path.getsize(filename)
                extra_args = {}
                guessed_type, guessed_encoding = guess_type(filename)
                if guessed_type is not None:
                    extra_args["ContentType"] = guessed_type
                if guessed_encoding is not None:
                    extra_args["ContentEncoding"] = guessed_encoding
                to_upload.append(
                    {
                        "Filename": filename,
                        "Bucket": s3_path.bucket,
                        "Key": posixpath.join(upload_path, f),
                        "ExtraArgs": extra_args,
                    }
                )
        state.total_num_files = total_num_files
        state.total_resource_size_bytes = total_file_bytes
        for upload_file_params in to_upload:
            s3_client.upload_file(
                **upload_file_params,
                Callback=state.increment_transferred_resource_size_bytes,
            )
            state.increment_num_transferred_files(1)

    @staticmethod
    def create_dir(
        credentials: AwsCredentials,
        bucket: str,
        key: str,
        *,
        endpoint_url: Optional[URL] = None,
    ) -> None:
        import boto3

        s3_kwargs = {"endpoint_url": endpoint_url and str(endpoint_url)}

        if credentials:
            s3_kwargs.update(
                {
                    "aws_access_key_id": credentials.access_key_id,
                    "aws_secret_access_key": credentials.secret_access_key,
                    "aws_session_token": credentials.session_token,
                }
            )

        s3_client = boto3.client("s3", **s3_kwargs)  # type: ignore
        s3_client.put_object(Bucket=bucket, Body="", Key=key)
