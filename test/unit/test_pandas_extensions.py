import io
from uuid import uuid4

import numpy as np
import pandas as pd
import PIL
import pytest
from pandas.testing import assert_frame_equal
from PIL.Image import Image

import layer
from layer.pandas_extensions import (
    Images,
    _ArrayDtype,
    _ImageDtype,
    _infer_custom_types,
)


_PARQUET_ENGINE = "pyarrow"


def _assert_image_columns_equal(left: pd.DataFrame, right: pd.DataFrame, col: str):
    # Cannot use assert_frame_equal, because of the incorrect comparison of boolean arrays in
    # https://github.com/pandas-dev/pandas/blob/ad190575aa75962d2d0eade2de81a5fe5a2e285b/pandas/_libs/testing.pyx#L177
    # due to ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    def to_numpy(d):
        return np.array([np.asarray(image) for image in d[col]])

    np.testing.assert_array_equal(to_numpy(left), to_numpy(right))


def test_pandas_images_read_write(tmp_path):
    data = _image_data_frame(num_images=3)

    parquet_path = tmp_path / str(uuid4())

    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    _assert_image_columns_equal(data, data_parquet, col="image")
    assert data_parquet["image"].dtype.name == "layer.image"


def test_pandas_images_head():
    data = _image_data_frame(num_images=32)
    assert_frame_equal(data.head(), _image_data_frame(num_images=5))
    assert_frame_equal(data.head(n=3), _image_data_frame(num_images=3))
    assert_frame_equal(data.head(n=-3), _image_data_frame(num_images=29))


@pytest.mark.parametrize(
    (
        "reduce_func",
        "result",
    ),
    # https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionArray._reduce.html
    [
        ("any", 0),
        ("all", 0),
        ("min", 0),
        ("max", 0),
        ("sum", 0),
        ("mean", 0),
        ("median", 0),
        ("prod", 0),
        ("std", 0),
        ("var", 0),
        ("sem", 0),
        ("kurt", 0),
        ("skew", 0),
    ],
)
def test_pandas_images_reduce(reduce_func, result):
    images = layer.Images([_image(n) for n in range(0, 3)])
    assert images._reduce(reduce_func) == result  # pylint: disable=protected-access


def test_pandas_images_info():
    data = _image_data_frame(num_images=42)
    with io.StringIO() as buf:
        data.info(buf=buf)
        info_str = buf.getvalue()
        # assert total bytes for each image is computed correctly
        assert "memory usage: 3.0 KB" in info_str, f"actual: {info_str}"


def test_seaborn_plot():
    try:
        import seaborn
    except ImportError:
        pytest.skip(msg="seaborn not installed")
    data = pd.DataFrame({"col": [1, 2, 42]})
    seaborn.histplot(data=data, x="col", color="green")


def test_pandas_images_eq_types_do_not_match():
    image0 = _image(0)
    image1 = _image(1)
    comp = Images(
        (
            image0,
            image1,
        )
    ).__eq__(1)
    assert comp.tolist() == [False, False]


def test_pandas_images_eq_equals():
    image0 = _image(0)
    image1 = _image(1)
    image2 = _image(0)
    image3 = _image(1)

    comp = Images((image0, image1,)).__eq__(
        Images(
            (
                image2,
                image3,
            )
        )
    )
    assert comp.tolist() == [True, True]


def test_pandas_images_eq_does_not_match():
    image0 = _image(0)
    image1 = _image(1)
    image2 = _image(0)
    image3 = _image(2)
    comp = Images((image0, image1,)).__eq__(
        Images(
            (
                image2,
                image3,
            )
        )
    )
    assert comp.tolist() == [True, False]


def test_pandas_images_eq_length_do_not_match_other():
    image0 = _image(0)
    image1 = _image(1)
    image2 = _image(0)
    comp = Images(
        (
            image0,
            image1,
        )
    ).__eq__(Images((image2,)))
    assert comp.tolist() == [True, False]


def test_pandas_images_eq_length_do_not_match_self():
    image0 = _image(0)
    image1 = _image(0)
    image2 = _image(1)
    comp = Images((image0,)).__eq__(
        Images(
            (
                image1,
                image2,
            )
        )
    )
    assert comp.tolist() == [True]


def test_pandas_images_concat_same_type():
    image0 = _image(0)
    image1 = _image(1)
    image2 = _image(2)

    images = Images._concat_same_type(  # pylint: disable=protected-access
        [
            (image0,),
            (
                image1,
                image2,
            ),
        ]
    )
    comp = Images((image0, image1, image2)).__eq__(images)

    assert comp.tolist() == [True, True, True]


def test_pandas_dropna_does_not_raise():
    # repro a bug where Images type is returned for every pandas type name
    df = pd.DataFrame({"col1": (1, 2, 3, 4)})
    df.dropna(how="all", inplace=True)


def test_pandas_images_construct_from_string_returns_imagedtype():
    assert isinstance(_ImageDtype.construct_from_string("layer.image"), _ImageDtype)


def test_pandas_images_construct_from_string_raises_type_error_for_other_types():
    with pytest.raises(
        TypeError, match=r"cannot construct a '_ImageDtype' from 'int64'"
    ):
        _ImageDtype.construct_from_string("int64")


def _image_data_frame(num_images: int) -> pd.DataFrame:
    return pd.DataFrame({"image": layer.Images([_image(n) for n in range(num_images)])})


def _image(n: int) -> Image:
    return PIL.Image.new("RGB", (1, 1), color=(73, 109, n))


def test_pandas_arrays_read_write(tmp_path):
    data = _array_data_frame()

    parquet_path = tmp_path / str(uuid4())

    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert_frame_equal(data, data_parquet)
    assert data_parquet["array"].dtype.name == "layer.ndarray"


def test_pandas_arrays_head():
    data = _array_data_frame()
    assert_frame_equal(data.head(), _array_data_frame())
    assert_frame_equal(data.head(n=3), _array_data_frame())


def test_pandas_arrays_info():
    data = _array_data_frame()
    with io.StringIO() as buf:
        data.info(buf=buf)
        info_str = buf.getvalue()
        # assert total bytes for each array is computed correctly
        assert "memory usage: 272.0 bytes" in info_str, f"actual: {info_str}"


def test_pandas_arrays_construct_from_string_returns_imagedtype():
    assert isinstance(_ArrayDtype.construct_from_string("layer.ndarray"), _ArrayDtype)


def test_pandas_arrays_construct_from_string_raises_type_error_for_other_types():
    with pytest.raises(
        TypeError, match=r"cannot construct a '_ArrayDtype' from 'int64'"
    ):
        _ArrayDtype.construct_from_string("int64")


def _array_data_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array": layer.Arrays(
                (
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
                    np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64),
                    np.array([[13, 14, 15], [16, 17, 18]], dtype=np.int64),
                )
            )
        }
    )


def test_pandas_images_describe():
    data = _image_data_frame(num_images=3)
    describe = pd.DataFrame(
        {"image": [3, 3, 0, 1]}, index=["count", "unique", "top", "freq"]
    )
    assert_frame_equal(describe, data.describe())


def test_pandas_arrays_describe():
    data = _array_data_frame()
    describe = pd.DataFrame(
        {"array": [3, 3, 0, 1]}, index=["count", "unique", "top", "freq"]
    )
    assert_frame_equal(describe, data.describe())


def test_infer_custom_types_returns_empty_data_frame_for_empty_data_frame():
    inferred = _infer_custom_types(pd.DataFrame())
    assert_frame_equal(pd.DataFrame(), inferred)


def test_infer_custom_types_returns_same_data_frame():
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["x", "y", "z"]})
    inferred = _infer_custom_types(data)
    assert_frame_equal(data, inferred)


def test_infer_custom_types_infers_images_type(tmp_path):
    data = pd.DataFrame({"col1": [1, 2, 3], "img": [_image(1), _image(2), _image(3)]})
    inferred = _infer_custom_types(data)
    parquet_path = tmp_path / str(uuid4())
    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert inferred["img"].dtype.name == "layer.image"
    # assert implictly converted types was written and read correctly
    _assert_image_columns_equal(data, data_parquet, col="img")


def test_infer_custom_types_infers_arrays_type(tmp_path):
    data = pd.DataFrame(
        {"col1": [1], "arr": [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)]}
    )
    inferred = _infer_custom_types(data)
    parquet_path = tmp_path / str(uuid4())
    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert inferred["arr"].dtype.name == "layer.ndarray"
    # assert implictly converted types was written and read correctly
    assert_frame_equal(data, data_parquet)


def test_infer_custom_types_does_not_infer_for_1dim_array(tmp_path):
    data = pd.DataFrame({"col1": [1], "arr": [np.array([1, 2, 3])]})
    inferred = _infer_custom_types(data)
    parquet_path = tmp_path / str(uuid4())
    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert inferred["arr"].dtype.name == "object"
    # assert implictly converted types was written and read correctly
    assert_frame_equal(data, data_parquet)
