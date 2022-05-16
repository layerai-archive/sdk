import io
from typing import Any, Generator, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import pyarrow as pa
from pandas._typing import PositionalIndexer
from pandas.core.arrays.base import ExtensionArray  # type: ignore

# TODO(donatasm): from pandas.core.dtypes.base import register_extension_dtype  # type: ignore
from pandas.core.dtypes.base import ExtensionDtype
from PIL.Image import Image


# TODO(donatasm): @register_extension_dtype
class _ImageDtype(ExtensionDtype):
    @property
    def name(self) -> str:
        return "layer.image"

    @classmethod
    def construct_from_string(cls, string: str) -> ExtensionDtype:
        return _ImageDtype()

    @classmethod
    def construct_array_type(cls) -> ExtensionArray:
        return Images

    @property
    def type(self) -> Image:
        return Image

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> "Images":
        if isinstance(array, pa.Array):
            chunks = array
        else:
            chunks = array.chunks
        images = (
            _image_from_binary_scalar(binary) for chunk in chunks for binary in chunk
        )
        return Images(tuple(images))


def _image_from_binary_scalar(binary: pa.BinaryScalar) -> Image:
    with io.BytesIO(binary.as_buffer().to_pybytes()) as buf:
        return PIL.Image.open(buf)


class Images(ExtensionArray):
    def __init__(self, images: Tuple[Image, ...]):
        self._images = images
        self._dtype = _ImageDtype()

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Optional[ExtensionDtype] = None,
        copy: bool = False,
    ) -> "Images":
        return Images(tuple(scalars))

    @property
    def dtype(self) -> ExtensionDtype:
        return self._dtype

    def copy(self) -> "Images":
        return Images(tuple(image.copy() for image in self._images))

    def isna(self) -> np.ndarray:  # type: ignore
        return np.array([image is not None for image in self._images], dtype=bool)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, item: PositionalIndexer) -> Union["Images", Image]:
        if isinstance(item, int):
            # for scalar item, return a scalar value suitable for the array's type
            return self._images[item]
        if isinstance(item, np.ndarray):
            # a boolean mask, filtered to the values where item is True
            return Images(tuple(self._get_images_by_mask(item)))
        raise NotImplementedError(f"item type {type(item)}")

    def _get_images_by_mask(self, mask: Iterable[bool]) -> Generator[Image, None, None]:
        for i, include in enumerate(mask):
            if include:
                yield self._images[i]

    def __arrow_array__(self, type: Any = None) -> pa.Array:
        return pa.array(self._images_byte_arr(), pa.binary())

    def _images_byte_arr(self) -> Generator[bytes, None, None]:
        for image in self._images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                yield buf.getvalue()
