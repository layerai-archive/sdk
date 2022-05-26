import io
import itertools
import warnings
from typing import Any, Generator, Iterable, Optional, Sequence, Type, Union

import numpy as np
import PIL.Image
import pyarrow as pa
from pandas._typing import PositionalIndexer  # type: ignore
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


_TYPE_IMAGE_NAME = "layer.image"


class _ImageType(pa.ExtensionType):
    """Arrow type extension https://arrow.apache.org/docs/python/extending_types.html#defining-extension-types-user-defined-types
    Provides addtional metadata for arrow schema about the image format.
    """

    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.binary(), _TYPE_IMAGE_NAME)

    def __arrow_ext_serialize__(self) -> bytes:
        return b"png"  # always store as png

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> pa.ExtensionType:
        return _ImageType()


class _ImageDtype(ExtensionDtype):
    @property
    def name(self) -> str:
        return _TYPE_IMAGE_NAME

    @classmethod
    def construct_from_string(cls, string: str) -> ExtensionDtype:
        if string == _TYPE_IMAGE_NAME:
            return _ImageDtype()
        raise TypeError(f"cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls) -> ExtensionArray:
        return Images

    @property
    def type(self) -> Type[Image]:
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
    def __init__(self, images: Sequence[Image]):
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
        return Images(scalars)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence["Images"]) -> "Images":
        return Images(tuple(itertools.chain(*to_concat)))

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
        eq_arr = np.empty(len(self), dtype=bool)
        eq_arr.fill(False)
        if not isinstance(other, Images):
            return eq_arr
        for i in range(min(len(self), len(other))):
            if _image_bytes(self._images[i]) == _image_bytes(other._images[i]):
                eq_arr[i] = True
        return eq_arr

    def __setitem__(
        self, key: Union[int, slice, "np.ndarray[Any, Any]"], value: Any
    ) -> None:
        raise NotImplementedError()

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


_TYPE_NDARRAY_NAME = "layer.ndarray"


class _ArrayType(pa.ExtensionType):
    """Arrow type extension https://arrow.apache.org/docs/python/extending_types.html#defining-extension-types-user-defined-types
    Provides addtional metadata for arrow schema about the ndarray format.
    """

    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.binary(), _TYPE_NDARRAY_NAME)

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> pa.ExtensionType:
        return _ArrayType()


class _ArrayDtype(ExtensionDtype):
    @property
    def name(self) -> str:
        return _TYPE_NDARRAY_NAME

    @classmethod
    def construct_from_string(cls, string: str) -> ExtensionDtype:
        if string == _TYPE_NDARRAY_NAME:
            return _ArrayDtype()
        raise TypeError(f"cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls) -> ExtensionArray:
        return Arrays

    @property
    def type(self) -> Type[np.ndarray]:  # type: ignore
        return np.ndarray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> "Arrays":
        if isinstance(array, pa.Array):
            chunks = array
        else:
            chunks = array.chunks
        arrays = (_load_array(binary) for chunk in chunks for binary in chunk)
        return Arrays(tuple(arrays))


def _load_array(binary: Union[pa.BinaryScalar, pa.ExtensionScalar]) -> np.ndarray:  # type: ignore
    storage_array = binary.value if isinstance(binary, pa.ExtensionScalar) else binary
    with io.BytesIO(storage_array.as_buffer().to_pybytes()) as buf:
        return np.load(buf, allow_pickle=False, fix_imports=False)  # type: ignore


class Arrays(ExtensionArray):
    def __init__(self, arrays: Sequence[np.ndarray]):  # type: ignore
        self._arrays = arrays
        self._dtype = _ArrayDtype()

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Optional[ExtensionDtype] = None,
        copy: bool = False,
    ) -> "Arrays":
        return Arrays(scalars)

    @property
    def dtype(self) -> ExtensionDtype:
        return self._dtype

    def copy(self) -> "Arrays":
        return Arrays(tuple(arr.copy() for arr in self._arrays))

    def isna(self) -> np.ndarray:  # type: ignore
        return np.array([arr is not None for arr in self._arrays], dtype=bool)

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, item: PositionalIndexer) -> Union["Arrays", np.ndarray]:  # type: ignore
        if isinstance(item, int):
            # for scalar item, return a scalar value suitable for the array's type
            return self._arrays[item]
        if isinstance(item, np.ndarray):
            # a boolean mask, filtered to the values where item is True
            return Arrays(tuple(self._get_arrays_by_mask(item)))
        if isinstance(item, slice):
            return Arrays(self._arrays[item.start : item.stop : item.step])
        raise NotImplementedError(f"item type {type(item)}")

    def _get_arrays_by_mask(
        self, mask: Iterable[bool]
    ) -> Generator[np.ndarray, None, None]:  # type: ignore
        for i, include in enumerate(mask):
            if include:
                yield self._arrays[i]

    def __arrow_array__(self, type: Any = None) -> pa.ExtensionArray:
        storage_array = pa.array(
            (_array_bytes(arr) for arr in self._arrays), pa.binary()
        )
        return pa.ExtensionArray.from_storage(_ArrayType(), storage_array)

    def __setitem__(
        self, key: Union[int, slice, "np.ndarray[Any, Any]"], value: Any
    ) -> None:
        raise NotImplementedError()

    @property
    def nbytes(self) -> int:
        total_bytes = 0
        for arr in self._arrays:
            total_bytes += arr.nbytes
        return total_bytes


def _array_bytes(arr: np.ndarray) -> bytes:  # type: ignore
    with io.BytesIO() as buf:
        np.save(buf, arr, allow_pickle=False, fix_imports=False)  # type: ignore
        return buf.getvalue()


def _register_type_extensions() -> None:
    # register arrow extension types
    pa.register_extension_type(_ImageType())
    pa.register_extension_type(_ArrayType())

    # register pandas extension types
    register_extension_dtype(_ImageDtype)
    register_extension_dtype(_ArrayDtype)
