import io
import warnings
from typing import Any, Generator, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import pyarrow as pa
from pandas._typing import PositionalIndexer
from pandas.core.arrays.base import ExtensionArray  # type: ignore
from pandas.core.dtypes.base import register_extension_dtype  # type: ignore
from pandas.core.dtypes.base import ExtensionDtype
from PIL.Image import Image


warnings.filterwarnings(
    "ignore",
    "The input object of type 'Image' is an array-like implementing one of the corresponding protocols.*",
)
warnings.filterwarnings(
    "ignore",
    "Creating an ndarray from ragged nested sequences.*",
)


class _ImageType(pa.ExtensionType):
    """Arrow type extension https://arrow.apache.org/docs/python/extending_types.html#defining-extension-types-user-defined-types
    Provides addtional metadata for arrow schema about the image format.
    """

    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.binary(), "layer.image")

    def __arrow_ext_serialize__(self) -> bytes:
        return b"png"  # always store as png

    @classmethod
    def __arrow_ext_deserialize__(
        self, storage_type: pa.DataType, serialized: bytes
    ) -> pa.ExtensionType:
        return _ImageType()


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
        images = (_load_image(binary) for chunk in chunks for binary in chunk)
        return Images(tuple(images))


def _load_image(binary: Union[pa.BinaryScalar, pa.ExtensionScalar]) -> Image:
    storage_array = binary.value if isinstance(binary, pa.ExtensionScalar) else binary
    with io.BytesIO(storage_array.as_buffer().to_pybytes()) as buf:
        image = PIL.Image.open(buf)
        image.load()
        return image


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
        if isinstance(item, slice):
            return Images(self._images[item.start : item.stop : item.step])
        raise NotImplementedError(f"item type {type(item)}")

    def _get_images_by_mask(self, mask: Iterable[bool]) -> Generator[Image, None, None]:
        for i, include in enumerate(mask):
            if include:
                yield self._images[i]

    def __arrow_array__(self, type: Any = None) -> pa.ExtensionArray:
        storage_array = pa.array(self._images_byte_arr(), pa.binary())
        # ignore the type and always read as _ImageType,
        # as it does not have any parameters for now
        return pa.ExtensionArray.from_storage(_ImageType(), storage_array)

    def _images_byte_arr(self) -> Generator[bytes, None, None]:
        for image in self._images:
            yield _image_bytes(image)

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs: Any) -> int:
        return 0

    def __eq__(self, other: Any) -> np.ndarray:  # type: ignore
        size = len(self._images)
        eq_arr = np.empty(size, dtype=bool)
        eq_arr.fill(False)
        if not isinstance(other, Images):
            return eq_arr
        for i in range(min(size, len(other))):
            if _image_bytes(self._images[i]) == _image_bytes(other._images[i]):
                eq_arr[i] = True
        return eq_arr

    @property
    def nbytes(self) -> int:
        total_bytes = 0
        for buf in self._images_byte_arr():
            total_bytes += len(buf)
        return total_bytes


def _image_bytes(image: Image) -> bytes:
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()


def _register_type_extensions() -> None:
    # register arrow extension types
    pa.register_extension_type(_ImageType())
    # register pandas extension types
    register_extension_dtype(_ImageDtype)
