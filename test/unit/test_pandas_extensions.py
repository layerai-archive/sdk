from uuid import uuid4

import pandas as pd
import PIL
import pytest
from pandas.testing import assert_frame_equal
from PIL import ImageDraw
from PIL.Image import Image

import layer


_PARQUET_ENGINE = "pyarrow"


@pytest.mark.skip(
    reason="until https://linear.app/layer/issue/LAY-3044/support-pandas-operations-for-images-class done"
)
def test_pandas_image_array_read_write(tmp_path):
    data = pd.DataFrame(
        {"image": layer.Images(tuple(generate_image(n) for n in range(0, 32)))}
    )

    parquet_path = tmp_path / str(uuid4())

    data.to_parquet(parquet_path, engine=_PARQUET_ENGINE)
    data_parquet = pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)

    assert_frame_equal(data, data_parquet)
    assert data_parquet["image"].dtype.name == "layer.image"


def generate_image(n: int) -> Image:
    image = PIL.Image.new("RGB", (160, 40), color=(73, 109, 137))
    draw = ImageDraw.Draw(image)
    draw.text((11, 10), f"Test #{n}", fill=(255, 255, 0))
    return image
