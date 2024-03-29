# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
# and IDIAP Research Institute nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas
import requests  # type: ignore

from ..logged_data.utils import get_base_module_list, has_allowed_extension


if TYPE_CHECKING:
    import matplotlib.axes._subplots  # type: ignore
    import matplotlib.figure  # type: ignore
    import numpy.typing as npt
    import PIL
    import torch


class Video:
    def __init__(
        self,
        video: "torch.Tensor",
        fps: Union[float, int] = 4,
    ):
        """
        :param video: Supported video types are:
        - torch.Tensor (tensor shape must be NTCHW or BNTCHW)
        :param fps: Frames per second
        """
        self.video = video
        self.fps = fps

    def get_video(self) -> Path:
        if "torch" in get_base_module_list(self.video):
            try:
                import torch

                assert isinstance(self.video, torch.Tensor)

                return Video._convert_to_video_path(self.video, self.fps)
            except ImportError:
                raise Exception(
                    "You need torch & torchvision installed to log torch.Tensor videos. Install with: `pip install torch torchvision`"
                )
        else:
            raise Exception(
                "Unsupported video type! Supported video types are: torch.Tensor"
            )

    # Inspired from https://github.com/pytorch/pytorch/blob/ed0091f8db1265449f13e2bdd1647bf873bd1fea/torch/utils/tensorboard/summary.py#L507
    @staticmethod
    def _convert_to_video_path(tensor: "torch.Tensor", fps: Union[float, int]) -> Path:
        tensor_np = Video._make_np(tensor)
        video_np = Video._prepare_video(tensor_np)
        # If user passes in uint8, then we don't need to rescale by 255
        scale_factor = Video._calc_scale_factor(video_np)
        video_np = video_np.astype(np.float32)
        video_np = (video_np * scale_factor).astype(np.uint8)
        video_path = Video._make_video(video_np, fps)
        return video_path

    @staticmethod
    def _make_video(tensor: np.ndarray, fps: Union[float, int]) -> Path:  # type: ignore
        try:
            import moviepy  # type: ignore # noqa pylint: disable=unused-import
        except ImportError:
            raise Exception(
                "You need moviepy installed to log torch.Tensor videos. Install with: `pip install moviepy`"
            )
        try:
            from moviepy import editor as mpy
        except ImportError:
            raise Exception(
                "moviepy is installed, but can't import moviepy.editor.",
                "Some packages could be missing [imageio, requests]",
            )

        # encode sequence of images into mp4 string
        clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

        filename = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        if TYPE_CHECKING:
            kwargs: Dict[str, Optional[bool]] = {}

        try:  # older versions of moviepy do not support logger argument
            kwargs = {"logger": None}
            clip.write_videofile(filename, **kwargs)
        except TypeError:
            try:  # even older versions of moviepy do not support progress_bar argument
                kwargs = {"verbose": False, "progress_bar": False}
                clip.write_videofile(filename, **kwargs)  # pylint: disable=E1123;
            except TypeError:
                kwargs = {
                    "verbose": False,
                }
                clip.write_videofile(filename, **kwargs)

        return Path(filename)

    @staticmethod
    def _calc_scale_factor(tensor: Union[np.ndarray, "torch.Tensor"]) -> int:  # type: ignore
        converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
        return 1 if converted.dtype == np.uint8 else 255

    @staticmethod
    def _prepare_video(video_tensor: np.ndarray) -> np.ndarray:  # type: ignore
        """
        Converts a 5D tensor [batchsize, time(frame), channel(color), height, width]
        into 4D tensor with dimension [time(frame), new_width, new_height, channel].
        A batch of images are spreaded to a grid, which forms a frame.
        e.g. Video with batchsize 16 will have a 4x4 grid.
        """

        if video_tensor.ndim < 4:
            raise ValueError("Video must be at least 4D: Time, Channels, Height, Width")
        elif video_tensor.ndim == 4:
            video_tensor = video_tensor.reshape(1, *video_tensor.shape)

        b, t, c, h, w = video_tensor.shape

        if video_tensor.dtype == np.uint8:
            video_tensor = np.float32(video_tensor) / 255.0  # type: ignore

        def is_power2(num: int) -> bool:
            return num != 0 and ((num & (num - 1)) == 0)

        # pad to nearest power of 2, all at once
        if not is_power2(video_tensor.shape[0]):
            len_addition = int(
                2 ** video_tensor.shape[0].bit_length() - video_tensor.shape[0]
            )
            video_tensor = np.concatenate(  # type: ignore
                (video_tensor, np.zeros(shape=(len_addition, t, c, h, w))), axis=0
            )

        n_rows = 2 ** ((b.bit_length() - 1) // 2)
        n_cols = video_tensor.shape[0] // n_rows

        video_tensor = np.reshape(video_tensor, newshape=(n_rows, n_cols, t, c, h, w))
        video_tensor = np.transpose(video_tensor, axes=(2, 0, 4, 1, 5, 3))
        video_tensor = np.reshape(video_tensor, newshape=(t, n_rows * h, n_cols * w, c))

        return video_tensor

    @staticmethod
    def _make_np(x: "torch.Tensor") -> np.ndarray:  # type: ignore
        """
        Args:
          x: An instance of torch tensor
        Returns:
            numpy.array: Numpy array
        """
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        raise NotImplementedError("Got {}, but torch tensor expected.".format(type(x)))

    @staticmethod
    def is_video(value: Any) -> bool:
        return isinstance(value, Video)


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
                img_tensors = self.img.clone()

                # Torchvision expects torch tensor image in CHW format, here we transpose the HWC array to match it
                if self.format == "HWC":
                    img_tensors = img_tensors.permute(2, 0, 1)

                return transforms.ToPILImage()(img_tensors)
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
    FILE = 9
    DIRECTORY = 10


@unique
class XCoordinateType(Enum):
    INVALID = 0
    STEP = 1
    TIME = 2


@dataclass(frozen=True)
class LoggedData:
    logged_data_type: LoggedDataType
    tag: str
    value: str
    values_with_coordinates: Dict[int, str]


@dataclass(frozen=True)
class Markdown:
    """
    Encapsulates a markdown text to be displayed correctly in Layer UI as part of logging.

    .. code-block:: python

        layer.log(layer.Markdown("## Hello world!"))

    """

    data: str


class LoggedDataObject:
    def __init__(self, logged_data: LoggedData, x_coordinate: Optional[int] = None):
        self._logged_data = logged_data
        self._x_coordinate = x_coordinate

    @staticmethod
    def _download_object_to(url: str, file_path: Path) -> None:
        response = requests.get(url)
        with open(file_path, "wb") as file_handler:
            file_handler.write(response.content)

    def value(self) -> Union[str, float, bool, pandas.DataFrame, "PIL.Image.Image"]:
        """
        To be called for string, numeric, boolean, image, markdown and table values.
        """
        if self.is_table():
            return pandas.read_json(self._logged_data.value, orient="table")
        elif self.is_number():
            return float(self._logged_data.value)
        elif self.is_text():
            return self._logged_data.value
        elif self.is_boolean():
            return bool(self._logged_data.value)
        elif self.is_image():
            from PIL.Image import open as pil_open

            url = (
                self._logged_data.value
                if self._x_coordinate is None
                else self._logged_data.values_with_coordinates[self._x_coordinate]
            )
            with tempfile.NamedTemporaryFile() as temp_file:
                self._download_object_to(url, Path(temp_file.name))
                image = pil_open(temp_file)
                image.load()
                return image
        elif self.is_markdown():
            return self._logged_data.value
        else:
            raise Exception(
                f"Use download_to() method for {self._logged_data.logged_data_type} type."
            )

    def download_to(self, path: Path) -> None:
        """
        To be called for file, video and directory logged data.
        """
        if self.is_blob() or self.is_video() or self.is_file():
            self._download_object_to(self._logged_data.value, path)
        elif self.is_directory():
            with tempfile.NamedTemporaryFile() as temp_file:
                self._download_object_to(self._logged_data.value, Path(temp_file.name))
                shutil.unpack_archive(temp_file.name, extract_dir=path, format="zip")
        else:
            raise Exception(
                f"Use value() method for {self._logged_data.logged_data_type} type."
            )

    def is_table(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.TABLE

    def is_blob(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.BLOB

    def is_number(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.NUMBER

    def is_text(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.TEXT

    def is_boolean(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.BOOLEAN

    def is_image(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.IMAGE

    def is_video(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.VIDEO

    def is_markdown(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.MARKDOWN

    def is_file(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.FILE

    def is_directory(self) -> bool:
        return self._logged_data.logged_data_type == LoggedDataType.DIRECTORY


LogValueType = Union[
    str,
    float,
    bool,
    int,
    List[Any],
    "np.ndarray[Any, Any]",
    Dict[str, Any],
    "pandas.DataFrame",
    "PIL.Image.Image",
    "matplotlib.figure.Figure",
    "matplotlib.axes._subplots.AxesSubplot",
    Image,
    ModuleType,
    Path,
    Markdown,
]

LogDataType = Mapping[
    str,
    LogValueType,
]
