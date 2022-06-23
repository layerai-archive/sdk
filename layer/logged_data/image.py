from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt

from .utils import get_base_module_list, has_allowed_extension


if TYPE_CHECKING:
    import PIL
    import tensorflow  # type: ignore
    import torch


class Image:
    def __init__(
        self,
        img: Union[
            "PIL.Image.Image",
            Path,
            npt.NDArray[np.complex64],
            "torch.Tensor",
            "tensorflow.Tensor",
        ],
        format: str = "CHW",
    ):
        self.img = img
        self.format = format

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
            if TYPE_CHECKING:
                import torch

                assert isinstance(self.img, torch.Tensor)
            try:
                from torchvision import transforms  # type: ignore

                return transforms.ToPILImage()(self.img)
            except ImportError:
                raise Exception(
                    "You need torchvision installed to log torch.Tensor images. Install with: `pip install torchvision`"
                )
        elif "tensorflow" in get_base_module_list(self.img):
            if TYPE_CHECKING:
                import tensorflow

                assert isinstance(self.img, tensorflow.Tensor)
            img_array = self.img.numpy()
            return self._get_image_from_array(img_array, self.format)
        else:
            raise Exception("Unsupported image type!")

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
        img_array: npt.NDArray[np.complex64], format: str
    ) -> "PIL.Image.Image":
        supported_image_formats = ["CHW", "HWC", "HW"]
        import PIL

        if format not in supported_image_formats:
            raise Exception(
                f"Invalid image format: '{format}'. Support formats are: {supported_image_formats}"
            )

        # Reshape array to HWC
        if format == "CHW":
            img_array = img_array.transpose(1, 2, 0)

        if format == "HW":
            img = PIL.Image.fromarray(np.uint8(img_array * 255), 'L')
        else:
            img_array = (img_array * 255).astype(np.uint8)
            img = PIL.Image.fromarray(img_array)

        return img
