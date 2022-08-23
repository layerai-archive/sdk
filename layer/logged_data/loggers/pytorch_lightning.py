import importlib.util
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, MutableMapping, Optional, Union

import pandas

import layer


if importlib.util.find_spec("pytorch_lightning") is None:

    class Logger:
        pass

else:
    from pytorch_lightning.loggers.logger import Logger  # type: ignore


class PytorchLightningLogger(Logger):
    r"""
        Log using `Layer <https://docs.app.layer.ai>`_.

    **Installation and set-up**

    Install with pip:

    .. code-block:: bash

        pip install layer

    Create a `Layer` instance:

    .. code-block:: python

        from pytorch_lightning.loggers import LayerLogger

        layer_logger = LayerLogger("[ORG/PROJECT_NAME]", "[API_KEY]")

    Pass the logger instance to the `Trainer`:

    .. code-block:: python

        trainer = Trainer(logger=layer_logger)

    Layer will log you in and init your project to track your experiment.

    **Log metrics**

    Log from :class:`~pytorch_lightning.core.module.LightningModule`:

    .. code-block:: python

        class LitMMNIST(LightningModule):
            def validation_step(self, batch, batch_idx):
                self.log("val_accuracy", acc)

    Use directly Layer:

    .. code-block:: python

        layer.log({"train/loss": loss}, step=5)

    **Log hyper-parameters**

    Save :class:`~pytorch_lightning.core.module.LightningModule` parameters:

    .. code-block:: python

        class LitMMNIST(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()


    **Log your model**

    Just return your model from your main training function. Layer will serialize/pickle your model and register to
    model catalog under your project.

    .. code-block:: python

        @layer.model("my_pl_model")
        def train():
            model = ...
            trainer = Trainer(...)
            trainer.fit(model, ...)
            return model


        train()


    **Log media**

    Log text with:

    .. code-block:: python

        # simple text
        layer_logger.log_text(key="dataset", text="mnist")

        # dictionary
        params = {"loss_type": "cross_entropy", "optimizer_type": "adamw"}
        layer_logger.log_metrics(params)

        # pandas DataFrame
        layer_logger.log_table(key="text", dataframe=df)

        # columns and data
        layer_logger.log_table(key="text", columns=["inp", "pred"], data=[["hllo", "hello"]])


    Log images with:

    .. code-block:: python

        # using tensors (`CHW`, `HWC`, `HW`), numpy arrays or PIL images
        layer_logger.log_image(key="image", image=img)

        # using file
        layer_logger.log_image(key="image", image="./pred.jpg")

        # add a step parameter to see images with slider
        layer_logger.log_image(key="image", image="./pred.jpg", step=epoch)

    Log videos with:

    .. code-block:: python

        # using tensors (`NTCHW`, `BNTCHW`)
        layer_logger.log_video(key="video", image=img)

        # using file
        layer_logger.log_video(key="video", image="./birds.mp4")


    See Also:
        - `Layer Pytorch Lightning Demo on Google Colab <https://bit.ly/pl_layer>`__
        - `Layer Documentation <https://docs.app.layer.ai>`__

    Args:
        project_name: Name of the Layer project
        api_key: Your Layer api key. You can call layer.login() if not provided.

    Raises:
        ModuleNotFoundError:
            If `layer` package is not installed.

    """

    PARAMETERS_KEY = "hyperparams"
    PREFIX_JOIN_CHAR = "-"

    def __init__(
        self,
        project_name: str,
        api_key: Optional[str] = None,
        prefix: str = "",
    ):
        super().__init__()

        self._project_name = project_name
        self._prefix = prefix
        self._api_key = api_key

        # Log user in
        if self._api_key is not None:
            layer.login_with_api_key(self._api_key)

        # Init project
        self._project = layer.init(self._project_name)

    @property
    def experiment(self) -> Any:
        r"""

        Top class `layer` object. To use Layer related functions, do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.any_layer_function(...)

        """

        return layer

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the project and the asset name.

        Returns:
            The name of the project and the model name
        """
        if self._context:
            return f"{self._project_name}/{self._context.asset_name()}"
        else:
            return None

    @property
    def version(self) -> Optional[Union[int, str]]:
        """Gets the full version of the model (eg. `2.3`)

        Returns:
            The model version in `[major].[minor]` format if training has started otherwise returns None
        """
        from layer.contracts.asset import AssetType

        if self._context and self._context.asset_type() == AssetType.MODEL:
            return f"{self._context.train().get_version()}.{self._context.train().get_train_index()}"
        else:
            return None

    @property
    def _context(self) -> Any:
        return layer.context.get_active_context()

    def log_text(self, key: str, text: str) -> None:
        """Log text.

        :param key: Name of the parameter/metric
        :param text: Parameter/Metric value
        :return:
        """
        self.experiment.log({key: text})

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """Log hyperparameters of your experiment.

        :param params: Hyperparameters key/values
        :return:
        """
        params = PytorchLightningLogger._convert_params(params)
        params = PytorchLightningLogger._flatten_dict(params)

        parameters_key = self.PARAMETERS_KEY

        self.experiment.log({parameters_key: params})

    def log_metrics(
        self,
        metrics: Mapping[
            str, Union[float, layer.Video, layer.Image, pandas.DataFrame, Path]
        ],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to your experiment.

        :param metrics: str/float key value pairs
        :param step: If provided, a chart will be generated from the logged float metrics
        :return:
        """
        metrics = PytorchLightningLogger._add_prefix(
            metrics, self._prefix, separator=self.PREFIX_JOIN_CHAR
        )
        self.experiment.log(dict(metrics), step=step)

    def log_image(
        self,
        key: str,
        image: Union[Any, Path],
        format: str = "CHW",
        step: Optional[int] = None,
    ) -> None:
        """Log an image to your experiment.

        :param key: Name of the image
        :param image: Image as `PIL.Image.Image`, `Path`, `npt.NDArray` or `torch.Tensor`
        :param format: Format of your array/tensor images. Can be: `CHW`, `HWC`, `HW`
        :param step: If provided, images for every step will be visible via a slider
        :return:
        """
        metrics = {key: layer.Image(image, format=format)}
        self.log_metrics(metrics, step)

    def log_video(
        self, key: str, video: Union[Any, Path], fps: Union[float, int] = 4
    ) -> None:
        """Log a video to your experiment.

        :param key: Name of your video
        :param video: Video as `torch.Tensor` in (`NTCHW`, `BNTCHW`) formats or `Path` of a video file
        :param fps: Frame per second, applicable to only torch tensor videos
        :return:
        """
        import torch

        if isinstance(video, torch.Tensor):
            self.log_metrics({key: layer.Video(video=video, fps=fps)})
        else:
            self.log_metrics({key: video})

    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Optional[Any] = None,
    ) -> None:
        """Log a table containing any object type (list, str, float, int, bool).

        :param key: Name of your table
        :param columns: Column names as list
        :param data: Rows as list
        :param dataframe: pandas Dataframe to be logged
        :return:
        """
        if dataframe is not None:
            self.log_metrics({key: dataframe})
        elif data is not None and columns is not None:
            df = pandas.DataFrame(columns=columns, data=data)
            self.log_metrics({key: df})
        else:
            raise Exception(
                "You should set either columns+data or dataframe parameter to log a table!"
            )

    # Taken from Pytorch Lightning Logger helpers
    # https://github.com/Lightning-AI/lightning
    @staticmethod
    def _add_prefix(
        metrics: Mapping[str, Any], prefix: str, separator: str
    ) -> Mapping[str, Any]:
        """Insert prefix before each key in a dict, separated by the separator."""
        if prefix:
            metrics = {f"{prefix}{separator}{k}": v for k, v in metrics.items()}

        return metrics

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        """Ensure parameters are a dict or convert to dict if necessary."""
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        return params

    def _flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:  # type: ignore
        """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``."""

        def _dict_generator(  # type: ignore
            input_dict: Any, prefixes: Any = None
        ) -> Generator[Any, Optional[List[str]], List[Any]]:
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, MutableMapping):
                for key, value in input_dict.items():
                    key = str(key)
                    if isinstance(value, (MutableMapping, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        yield from _dict_generator(value, prefixes + [key])
                    else:
                        yield prefixes + [
                            key,
                            value if value is not None else str(None),
                        ]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}
