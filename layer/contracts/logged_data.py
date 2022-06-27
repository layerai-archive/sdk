from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np

from ..logged_data.utils import get_base_module_list, has_allowed_extension


if TYPE_CHECKING:
    import numpy.typing as npt
    import PIL
    import torch


class Image:
    """
    Helper class to log complex images such as torch.tensor or numpy.ndarray

    Example of logging a numpy array as an image:
    .. code-block:: python
        ...
        import numpy as np
        img_HWC = np.zeros((100, 100, 3))
        img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

        layer.log({"image":layer.Image(img_HWC)})
        ...
    """

    def __init__(
        self,
        img: Union[
            "PIL.Image.Image", Path, "npt.NDArray[np.complex64]", "torch.Tensor"
        ],
        format: str = "CHW",
    ):
        """
        :param img: Supported image types are:
        - torch.Tensor
        - numpy.ndarray
        - Image from path.Path
        - PIL.Image
        :param format: Sets the image format if the provided image is a numpy.ndarray.
        Support formats are: `CHW`, `HWC`, `HW`
        """
        self.img = img
        self.format = format
        supported_image_formats = ["CHW", "HWC", "HW"]

        if format not in supported_image_formats:
            raise ValueError(
                f"Invalid image format: '{format}'. Support formats are: {supported_image_formats}"
            )

    def get_image(self) -> Union["PIL.Image.Image", Path]:
        if Image.is_pil_image(self.img):
            if TYPE_CHECKING:
                import PIL

                assert isinstance(self.img, PIL.Image.Image)
            return self.img
        elif Image.is_image_path(self.img):
            if TYPE_CHECKING:
                assert isinstance(self.img, Path)
            return self.img
        elif isinstance(self.img, np.ndarray):
            return Image._get_image_from_array(self.img, self.format)
        elif "torch" in get_base_module_list(self.img):
            try:
                import torch
                from torchvision import transforms  # type: ignore

                assert isinstance(self.img, torch.Tensor)

                return transforms.ToPILImage()(self.img)
            except ImportError:
                raise Exception(
                    "You need torch & torchvision installed to log torch.Tensor images. Install with: `pip install torch torchvision`"
                )
        else:
            raise Exception(
                "Unsupported image type! Supported images are: PIL.Image, torch.Tensor, np.ndarray (CHW,HWC,HW) and Path"
            )

    @staticmethod
    def is_image(value: Any) -> bool:
        return (
            isinstance(value, Image)
            or Image.is_pil_image(value)
            or Image.is_image_path(value)
        )

    @staticmethod
    def is_pil_image(value: Any) -> bool:
        return "PIL.Image" in get_base_module_list(value)

    @staticmethod
    def is_image_path(path: Any) -> bool:
        return isinstance(path, Path) and has_allowed_extension(
            path, [".gif", ".png", ".jpg", ".jpeg"]
        )

    @staticmethod
    def _get_image_from_array(
        img_array: "npt.NDArray[np.complex64]", format: str
    ) -> "PIL.Image.Image":
        from PIL import Image as PIL_IMAGE

        # Reshape array to HWC
        if format == "CHW":
            img_array = img_array.transpose(1, 2, 0)

        # Users can pass [0, 1] (float32) or [0, 255] (uint8), we should scale accordingly
        scale_factor = 1 if img_array.dtype == np.uint8 else 255  # type: ignore

        if format == "HW":
            img = PIL_IMAGE.fromarray(np.uint8(img_array * scale_factor), "L")
        else:
            img_array = (img_array * scale_factor).astype(np.uint8)
            img = PIL_IMAGE.fromarray(img_array)

        return img


@unique
class LoggedDataType(Enum):
    INVALID = 0
    TEXT = 1
    TABLE = 2
    BLOB = 3
    NUMBER = 4
    BOOLEAN = 5
    IMAGE = 6
    VIDEO = 7
    MARKDOWN = 8


@dataclass(frozen=True)
class LoggedData:
    logged_data_type: LoggedDataType
    tag: str
    data: str
    epoched_data: Dict[int, str]


@dataclass(frozen=True)
class ModelMetricPoint:
    epoch: int
    value: float


@dataclass(frozen=True)
class Markdown:
    """
    Encapsulates a markdown text to be displayed correctly in Layer UI as part of logging.

    .. code-block:: python

        layer.log(layer.Markdown("## Hello world!"))

    """

    data: str
