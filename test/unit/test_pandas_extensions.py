import io
from uuid import uuid4

import pandas as pd
import PIL
import pytest
from pandas.testing import assert_frame_equal
from PIL import ImageDraw
from PIL.Image import Image

import layer
from layer.pandas_extensions import Images


_PARQUET_ENGINE = "pyarrow"


def test_pandas_images_read_write(tmp_path):
    data = _image_data_frame(num_images=32)

    parquet_path = tmp_path / str(uuid4())

    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert_frame_equal(data, data_parquet)
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
    images = layer.Images([_generate_image(n) for n in range(0, 3)])
    assert images._reduce(reduce_func) == result


def test_pandas_images_info():
    data = _image_data_frame(num_images=42)
    with io.StringIO() as buf:
        data.info(buf=buf)
        info_str = buf.getvalue()
        # assert total bytes for each image is computed correctly
        assert "memory usage: 14.1 KB" in info_str, f"actual: {info_str}"


def test_seaborn_plot():
    try:
        import seaborn
    except ImportError:
        pytest.skip(msg="seaborn not installed")
    data = pd.DataFrame({"col": [1, 2, 42]})
    seaborn.histplot(data=data, x="col", color="green")


def test_images_eq_types_do_not_match():
    image0 = _generate_image(0)
    image1 = _generate_image(1)
    comp = Images(
        (
            image0,
            image1,
        )
    ).__eq__(1)
    assert comp.tolist() == [False, False]


def test_images_eq_equals():
    image0 = _generate_image(0)
    image1 = _generate_image(1)
    image2 = _generate_image(0)
    image3 = _generate_image(1)

    comp = Images((image0, image1,)).__eq__(
        Images(
            (
                image2,
                image3,
            )
        )
    )
    assert comp.tolist() == [True, True]


def test_images_eq_does_not_match():
    image0 = _generate_image(0)
    image1 = _generate_image(1)
    image2 = _generate_image(0)
    image3 = _generate_image(2)
    comp = Images((image0, image1,)).__eq__(
        Images(
            (
                image2,
                image3,
            )
        )
    )
    assert comp.tolist() == [True, False]


def test_images_eq_length_do_not_match_other():
    image0 = _generate_image(0)
    image1 = _generate_image(1)
    image2 = _generate_image(0)
    comp = Images(
        (
            image0,
            image1,
        )
    ).__eq__(Images((image2,)))
    assert comp.tolist() == [True, False]


def test_images_eq_length_do_not_match_self():
    image0 = _generate_image(0)
    image1 = _generate_image(0)
    image2 = _generate_image(1)
    comp = Images((image0,)).__eq__(
        Images(
            (
                image1,
                image2,
            )
        )
    )
    assert comp.tolist() == [True]


def test_concat_same_type():
    image0 = _generate_image(0)
    image1 = _generate_image(1)
    image2 = _generate_image(2)

    images = Images._concat_same_type(
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


def _image_data_frame(num_images: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"image": layer.Images([_generate_image(n) for n in range(num_images)])}
    )


def _generate_image(n: int) -> Image:
    image = PIL.Image.new("RGB", (160, 40), color=(73, 109, 137))
    draw = ImageDraw.Draw(image)
    draw.text((11, 10), f"Test #{n}", fill=(255, 255, 0))
    return image
