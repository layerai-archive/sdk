import io
from uuid import uuid4

import pandas as pd
import PIL
import pytest
from pandas.testing import assert_frame_equal
from PIL import ImageDraw
from PIL.Image import Image

import layer


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
    images = layer.Images(tuple(_generate_image(n) for n in range(0, 3)))
    assert images._reduce(reduce_func) == result


def test_pandas_images_info():
    data = _image_data_frame(num_images=42)
    with io.StringIO() as buf:
        data.info(buf=buf)
        info_str = buf.getvalue()
        # assert total bytes for each image is computed correctly
        assert "memory usage: 14.1 KB" in info_str, f"actual: {info_str}"


def _image_data_frame(num_images: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"image": layer.Images(tuple(_generate_image(n) for n in range(num_images)))}
    )


def _generate_image(n: int) -> Image:
    image = PIL.Image.new("RGB", (160, 40), color=(73, 109, 137))
    draw = ImageDraw.Draw(image)
    draw.text((11, 10), f"Test #{n}", fill=(255, 255, 0))
    return image
