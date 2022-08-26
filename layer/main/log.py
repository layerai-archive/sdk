import logging
from typing import Optional

from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.context import get_active_context
from layer.contracts.logged_data import LogDataType
from layer.logged_data.log_data_runner import LogDataRunner
from layer.utils.async_utils import asyncio_run_in_thread

from ..logged_data.immediate_logged_data_destination import (
    ImmediateLoggedDataDestination,
)
from .utils import sdk_function


logger = logging.getLogger(__name__)


@sdk_function
def log(
    data: LogDataType,
    step: Optional[int] = None,
    category: Optional[str] = None,
) -> None:
    """
    :param data: A dictionary in which each key is a string tag (i.e. name/id). The value can have different types. See examples below for more details.
    :param step: An optional non-negative integer that associates data with a particular step (epoch). This only takes effect if the logged data is to be associated with a model train (and *not* with a dataset build), and the data is either a number or an image.
    :param category: An optional string that associates data with a particular category. This category is used for grouping in the web UI.
    :return: None

    Logs arbitrary data associated with a model train or a dataset build into Layer backend.

    This function can only be run inside functions decorated with ``@model`` or ``@dataset``. The logged data can then be discovered and analyzed through the Layer UI.

    We support Python primitives, images, tables and plots to enable better experiment tracking and version comparison.

    **Python Primitives**

    You can log Python primitive types for your model train parameters, metrics or KPIs

    Accepted Types:
    ``str``,
    ``float``,
    ``bool``,
    ``int``

    **List Types**

    These are converted into string and stored as text.

    ``list``
    ``np.ndarray`` (i.e. a normal NumPy array)

    **Markdown**

    You can put markdown syntax and it will be rendered in the web UI accordingly.

    Accepted Types:
    ``layer.Markdown``

    **Images**

    You can log images to track inputs, outputs, detections, activations and more. We support GIF, JPEG, PNG formats.

    Accepted Types:
    ``PIL.Image.Image``,
    ``path.Path``
    ``layer.Image``

    **Videos**

    We support MP4, WebM, Ogg file formats or ``pytorch.Tensor`` (with BNTCHW+NTCHW shape) through ``layer.Video``.

    Accepted Types:
    ``path.Path``
    ``layer.Video``

    **Files and Directories**

    Directories and files (that do not have an image or video extension, as documented above) can also be logged.

    If a directory is passed, it is compressed and logged as a single file.

    Accepted Types:
    ``path.Path``

    **Charts**

    You can track your metrics in detail with charts. Metrics logged with the same layer.log(...) call will be
    grouped and visualized together in the web UI.

    Accepted Types:
    ``matplotlib.figure.Figure``,
    ``matplotlib.pyplot``,
    ``matplotlib.axes._subplots.AxesSubplot``,
    ``ModuleType`` (only for the matplotlib module, for convenience)

    **Tables**

    You can log dataframes to display and analyze your tabular data.

    Accepted Types:
    ``pandas.DataFrame``
    ``dict`` (the key should be a string. The value either needs to be a primitive type or it will be converted to str)

    .. code-block:: python

        import layer
        import matplotlib.pyplot as plt
        import pandas as pd
        from layer.decorators import dataset, model

        # Define a new function for dataset generation
        @dataset("my_dataset_name")
        def create_my_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])

            t = np.arange(0.0, 2.0, 0.01)
            s = 1 + np.sin(2 * np.pi * t)
            fig, ax = plt.subplots()

            layer.log({
                "my-str-tag": "any text",
                "my-int-tag": 123, # any number
                "foo-bool": True,
                "some-sample-dataframe-tag": ..., # Pandas data frame
                "some-local-image-file": Path.home() / "images/foo.png",
                "some-matplot-lib-figure": fig, # You could alternatively just pass plt as well, and Layer would just get the current/active figure.
            })

            layer.log({
                "another-tag": "you can call layer.log multiple times"
            })

            return dataframe

        @model("my_model")
        def create_my_model():
            # everything that can be logged for a dataset (as shown above), can also be logged for a model too.

            # In addition to those, if you have a multi-step (i.e. multi-epoch) algorithm, you can associate a metric with the step.
            # These will be rendered on a graph inside the Layer UI.
            for i in range(1000):
                some_result, accuracy = some_algo(some_result)
                layer.log({
                    "Model accuracy": accuracy,
                }, step=i)

        create_my_dataset()
        create_my_model()
    """
    active_context = get_active_context()
    if not active_context:
        raise RuntimeError(
            "Data logging only allowed inside functions either decorated with @layer.model or @layer.dataset or"
            "invoked programmatically via layer.model('model-name')(my_function)(*my_function_arguments)"
        )
    train = active_context.train()
    train_id = train.get_id() if train is not None else None
    dataset_build = active_context.dataset_build()
    dataset_build_id = dataset_build.id if dataset_build is not None else None
    layer_config = asyncio_run_in_thread(ConfigManager().refresh())

    with LayerClient(layer_config.client, logger).init() as client:
        logged_data_destination = active_context.logged_data_destination()
        if not logged_data_destination:
            # temporary fallback: in the future, this will be new queuing destination,
            # but if it's not yet passed to the context, use old logic
            logged_data_destination = ImmediateLoggedDataDestination(
                client.logged_data_service_client
            )

        log_data_runner = LogDataRunner(
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            logger=logger,
            logged_data_destination=logged_data_destination,
        )
        log_data_runner.log(data=data, x_coordinate=step, category=category)
