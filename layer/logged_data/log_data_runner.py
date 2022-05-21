import inspect
import io
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

import pandas as pd
import requests  # type: ignore

from layer.clients.layer import LayerClient
from layer.contracts.logged_data import ModelMetricPoint


if TYPE_CHECKING:
    import matplotlib.figure  # type: ignore
    import PIL.Image


class LogDataRunner:
    def __init__(
        self,
        client: LayerClient,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        logger: Optional[Logger] = None,
    ):
        assert bool(train_id) ^ bool(dataset_build_id)
        self._client = client
        self._train_id = train_id
        self._dataset_build_id = dataset_build_id
        self._logger = logger

    def log(
        self,
        data: Dict[
            str,
            Union[
                str,
                float,
                bool,
                int,
                pd.DataFrame,
                "PIL.Image.Image",
                "matplotlib.figure.Figure",
                ModuleType,
                Path,
            ],
        ],
        epoch: Optional[int] = None,
    ) -> None:
        for tag, value in data.items():
            if isinstance(value, str):
                self._log_text(tag=tag, text=value)
            # boolean check must be done before numeric check as it also returns true for booleans.
            elif isinstance(value, bool):
                self._log_boolean(tag=tag, bool_val=value)
            elif isinstance(value, (int, float)):
                if self._train_id and epoch is not None:
                    self._log_metric(tag=tag, numeric_value=value, epoch=epoch)
                else:
                    self._log_number(tag=tag, number=value)
            elif isinstance(value, pd.DataFrame):
                self._log_dataframe(tag=tag, df=value)
            elif isinstance(value, Path):
                self._log_image_from_path(tag=tag, path=value)
            elif self._is_pil_image(value):
                if TYPE_CHECKING:
                    import PIL.Image

                    assert isinstance(value, PIL.Image.Image)
                self._log_image(tag=tag, image=value)
            elif self._is_plot_figure(value):
                if TYPE_CHECKING:
                    import matplotlib.figure

                    assert isinstance(value, matplotlib.figure.Figure)
                self._log_plot_figure(tag=tag, figure=value)
            elif self._is_pyplot(value):
                assert isinstance(value, ModuleType)
                self._log_current_plot_figure(tag=tag, plt=value)
            else:
                raise ValueError(f"Unsupported value type -> {type(value)}")

    def _log_metric(
        self, tag: str, numeric_value: Union[float, int], epoch: Optional[int]
    ) -> None:
        assert self._train_id
        # store numeric values w/o an explicit epoch as metric with the special epoch:-1
        epoch = epoch if epoch is not None else -1
        self._client.logged_data_service_client.log_model_metric(
            train_id=self._train_id,
            tag=tag,
            points=[ModelMetricPoint(epoch=epoch, value=float(numeric_value))],
        )

    def _log_text(self, tag: str, text: str) -> None:
        self._client.logged_data_service_client.log_text_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=text,
        )

    def _log_number(self, tag: str, number: Union[float, int]) -> None:
        self._client.logged_data_service_client.log_numeric_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=str(number),
        )

    def _log_boolean(self, tag: str, bool_val: bool) -> None:
        self._client.logged_data_service_client.log_boolean_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=str(bool_val),
        )

    def _log_dataframe(self, tag: str, df: pd.DataFrame) -> None:
        rows = len(df.index)
        if rows > 1000:
            raise ValueError(
                f"DataFrame rows size cannot exceed 1000. Current size: {rows}"
            )
        df_json = df.to_json(orient="table")  # type: ignore
        self._client.logged_data_service_client.log_table_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=df_json,
        )

    def _log_image_from_path(self, tag: str, path: Path) -> None:
        file_size_in_bytes = path.stat().st_size
        self._check_size_less_than_1_mb(file_size_in_bytes)
        with requests.Session() as s, open(path, "rb") as image_file:
            presigned_url = self._client.logged_data_service_client.log_binary_data(
                train_id=self._train_id,
                dataset_build_id=self._dataset_build_id,
                tag=tag,
            )
            resp = s.put(presigned_url, data=image_file)
            resp.raise_for_status()

    def _log_image(self, tag: str, image: "PIL.Image.Image") -> None:
        with requests.Session() as s, io.BytesIO() as buffer:
            if image.mode in ["RGBA", "P"]:
                image.save(buffer, format="PNG")
            else:
                image.save(buffer, format="JPEG")
            self._check_buffer_size(buffer=buffer)
            presigned_url = self._client.logged_data_service_client.log_binary_data(
                train_id=self._train_id,
                dataset_build_id=self._dataset_build_id,
                tag=tag,
            )
            resp = s.put(presigned_url, data=buffer.getvalue())
            resp.raise_for_status()

    def _log_plot_figure(self, tag: str, figure: "matplotlib.figure.Figure") -> None:
        with requests.Session() as s, io.BytesIO() as buffer:
            figure.savefig(buffer, format="jpg")
            self._check_buffer_size(buffer=buffer)
            presigned_url = self._client.logged_data_service_client.log_binary_data(
                train_id=self._train_id,
                dataset_build_id=self._dataset_build_id,
                tag=tag,
            )
            resp = s.put(presigned_url, data=buffer.getvalue())
            resp.raise_for_status()

    def _log_current_plot_figure(self, tag: str, plt: ModuleType) -> None:
        if len(plt.get_fignums()) > 0:
            self._log_plot_figure(tag, plt.gcf())
        else:
            raise ValueError("No figures in the current pyplot state!")

    def _is_pil_image(self, value: Any) -> bool:
        return "PIL.Image" in self._get_base_module_list(value)

    def _is_plot_figure(self, value: Any) -> bool:
        return "matplotlib.figure" in self._get_base_module_list(value)

    def _is_pyplot(self, value: Any) -> bool:
        return (
            hasattr(value, "__name__")
            and "matplotlib.pyplot" == value.__name__
            and isinstance(value, ModuleType)
        )

    def _get_base_module_list(self, value: Any) -> List[str]:
        return [
            inspect.getmodule(clazz).__name__ for clazz in inspect.getmro(type(value))  # type: ignore
        ]

    def _check_buffer_size(self, buffer: io.BytesIO) -> None:
        size_in_bytes = buffer.tell()
        self._check_size_less_than_1_mb(size_in_bytes)

    def _check_size_less_than_1_mb(self, size_in_bytes: float) -> None:
        size_in_mb = size_in_bytes / 1000**2
        if size_in_mb > 1:
            raise ValueError(
                f"Image size cannot exceed 1MB. Current size: {size_in_mb}"
            )
