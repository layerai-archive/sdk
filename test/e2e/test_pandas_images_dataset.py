import pandas as pd
import PIL
from pandas.testing import assert_frame_equal
from PIL import ImageDraw
from PIL.Image import Image

import layer
from layer.contracts.projects import Project
from layer.decorators import dataset, pip_requirements
from test.e2e.assertion_utils import E2ETestAsserter


def test_pandas_images_dataset_store_and_save(
    initialized_project: Project, asserter: E2ETestAsserter
):
    @dataset("images")
    @pip_requirements(packages=["Pillow==9.1.1"])
    def build_images() -> pd.DataFrame:
        def _generate_image(n: int) -> Image:
            image = PIL.Image.new("RGB", (160, 40), color=(73, 109, 137))
            draw = ImageDraw.Draw(image)
            draw.text((11, 10), f"Test #{n}", fill=(255, 255, 0))
            return image

        return pd.DataFrame(
            {"image": layer.Images(tuple(_generate_image(n) for n in range(8)))}
        )

    run = layer.run([build_images])
    asserter.assert_run_succeeded(run.id)

    data = layer.get_dataset(f"{initialized_project.name}/datasets/images").to_pandas()

    assert_frame_equal(build_images(), data)
