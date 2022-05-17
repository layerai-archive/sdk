import re
import unicodedata

from layerapi.api.entity.model_version_pb2 import (  # pylint: disable=unused-import
    ModelVersion,
)
from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.value.aws_credentials_pb2 import AwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path


class ModelDefinition:
    """Holds information regarding an ML model.

    This class holds structural information about an ML Model and is able to construct a path where
    you can find the model in the storage. It also stores the metadata associated with the corresponding model train.
    """

    def __init__(
        self,
        name: str,
        train_id: ModelTrainId,
        proto_flavor: "ModelVersion.ModelFlavor.V",
        s3_path: S3Path,
        credentials: AwsCredentials,
    ):
        self.__model_name: str = slugify(name).replace("-", "")
        self.__train_id: ModelTrainId = train_id
        self.__proto_flavor: "ModelVersion.ModelFlavor.V" = proto_flavor
        self.__s3_path: S3Path = s3_path
        self.__credentials: AwsCredentials = credentials
        self.__raw_name: str = name

    @property
    def model_name(self) -> str:
        """Returns the model name

        Returns:
            A string name, as it is in the stored metadata
        """
        return self.__model_name

    @property
    def model_raw_name(self) -> str:
        """Returns the model name with no sluggification

        Returns:
            A string name
        """
        return self.__raw_name

    @property
    def model_train_id(self) -> ModelTrainId:
        """Returns the model train id"""
        return self.__train_id

    @property
    def proto_flavor(self) -> "ModelVersion.ModelFlavor.V":
        """Returns the proto flavor

        Returns:
            A string - the proto flavor, used to infer the type of the model obj to instantiate
        """
        return self.__proto_flavor

    @property
    def s3_path(self) -> S3Path:
        """Returns the s3 path where the model is stored

        Returns:
            A S3Path proto - The s3 path where the model is stored
        """
        return self.__s3_path

    @property
    def credentials(self) -> AwsCredentials:
        """Returns the credentials object for this model

        Returns:
            A AwsCredentials proto - The credentials object for this model
        """
        return self.__credentials

    def __repr__(self) -> str:
        return (
            f"ModelDefinition{{"
            f"model_name:{self.model_name}"
            f"model_train_id:{self.model_train_id.value}"
            f"proto_flavor:{self.proto_flavor}"
            f"s3_path:{self.s3_path}"
            f"}}"
        )


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.

    Taken unchanged from django/utils/text.py
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)
