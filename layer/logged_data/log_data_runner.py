import io
import shutil
import tempfile
import uuid
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
import requests  # type: ignore
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType

from layer.clients.layer import LayerClient
from layer.contracts.logged_data import Image, Markdown, ModelMetricPoint, Video

from .utils import get_base_module_list, has_allowed_extension


if TYPE_CHECKING:
    import matplotlib.figure  # type: ignore
    import PIL


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

    def log(  # pylint: disable=too-many-statements
        self,
        data: Dict[
            str,
            Union[
                str,
                float,
                bool,
                int,
                List[Any],
                "np.ndarray[Any, Any]",
                Dict[str, Any],
                pd.DataFrame,
                "PIL.Image.Image",
                "matplotlib.figure.Figure",
                Image,
                ModuleType,
                Path,
                Markdown,
            ],
        ],
        epoch: Optional[int] = None,
        category: Optional[str] = None,
        grouped: Optional[bool] = False,
    ) -> None:
        LogDataRunner._check_epoch(epoch)

        common_metric_group_uuid = uuid.uuid4()
        for tag, value in data.items():
            if isinstance(value, str):
                self._log_text(tag=tag, text=value, category=category)
            elif isinstance(value, list):
                self._log_text(tag=tag, text=str(value), category=category)
            elif isinstance(value, np.ndarray):
                self._log_text(tag=tag, text=str(value.tolist()), category=category)
            # boolean check must be done before numeric check as it also returns true for booleans.
            elif isinstance(value, bool):
                self._log_boolean(tag=tag, bool_val=value, category=category)
            elif isinstance(value, (int, float)):
                if self._train_id and epoch is not None:
                    self._log_metric(
                        tag=tag,
                        numeric_value=value,
                        epoch=epoch,
                        metric_group_id=common_metric_group_uuid
                        if grouped
                        else uuid.uuid4(),
                        category=category,
                    )
                else:
                    self._log_number(tag=tag, number=value, category=category)
            elif isinstance(value, Markdown):
                self._log_markdown(tag=tag, text=value.data, category=category)
            elif isinstance(value, pd.DataFrame):
                self._log_dataframe(tag=tag, df=value, category=category)
            elif LogDataRunner._is_tabular_dict(value):
                if TYPE_CHECKING:
                    assert isinstance(value, dict)
                dataframe = LogDataRunner._convert_dict_to_dataframe(value)
                self._log_dataframe(tag=tag, df=dataframe, category=category)
            elif LogDataRunner._is_video_from_path(value):
                if TYPE_CHECKING:
                    assert isinstance(value, Path)
                self._log_video_from_path(tag=tag, path=value, category=category)
            elif isinstance(value, Video):
                video_path = value.get_video()
                self._log_video_from_path(tag=tag, path=video_path, category=category)
            elif Image.is_image(value):
                if TYPE_CHECKING:
                    import PIL

                    assert isinstance(value, (Path, PIL.Image.Image, Image))
                self._log_image(
                    tag=tag, img_value=value, epoch=epoch, category=category
                )
            elif self._is_plot_figure(value):
                if TYPE_CHECKING:
                    import matplotlib.figure

                    assert isinstance(value, matplotlib.figure.Figure)
                self._log_plot_figure(tag=tag, figure=value, category=category)
            elif self._is_axes_subplot(value):
                if TYPE_CHECKING:
                    import matplotlib.axes._subplots  # type: ignore

                    assert isinstance(
                        value,
                        matplotlib.axes._subplots.AxesSubplot,  # pylint: disable=protected-access
                    )
                self._log_plot_figure(
                    tag=tag, figure=value.get_figure(), category=category
                )
            elif self._is_pyplot(value):
                assert isinstance(value, ModuleType)
                self._log_current_plot_figure(tag=tag, plt=value, category=category)
            elif isinstance(value, Path):  # This must be below image and video!
                self._log_file_or_directory_from_path(
                    tag=tag, path=value, category=category
                )
            else:
                raise ValueError(f"Unsupported value type -> {type(value)}")

    def _log_image(
        self,
        tag: str,
        img_value: Union["PIL.Image.Image", Path, Image],
        epoch: Optional[int],
        category: Optional[str] = None,
    ) -> None:
        # Users can log images directly with `layer.log({'img':img})` for simple images or with
        # `layer.log({'img':layer.Image(img)})` for advanced image formats like Tensors
        if isinstance(img_value, Image):
            img = img_value.get_image()
        else:
            if TYPE_CHECKING:
                import PIL

                assert isinstance(img_value, Path) or isinstance(
                    img_value, PIL.Image.Image
                )
            img = img_value

        if Image.is_image_path(img):
            if TYPE_CHECKING:
                assert isinstance(img, Path)
            if self._train_id and epoch is not None:
                self._log_image_from_path(
                    tag=tag, path=img, epoch=epoch, category=category
                )
            else:
                self._log_image_from_path(tag=tag, path=img, category=category)
        elif Image.is_pil_image(img):
            if TYPE_CHECKING:
                import PIL

                assert isinstance(img, PIL.Image.Image)
            if self._train_id and epoch is not None:
                self._log_pil_image(tag=tag, image=img, epoch=epoch, category=category)
            else:
                self._log_pil_image(tag=tag, image=img, category=category)

    def _log_metric(
        self,
        tag: str,
        numeric_value: Union[float, int],
        epoch: Optional[int],
        metric_group_id: UUID,
        category: Optional[str] = None,
    ) -> None:
        assert self._train_id
        # store numeric values w/o an explicit epoch as metric with the special epoch:-1
        epoch = epoch if epoch is not None else -1
        self._client.logged_data_service_client.log_model_metric(
            train_id=self._train_id,
            tag=tag,
            points=[ModelMetricPoint(epoch=epoch, value=float(numeric_value))],
            metric_group_id=metric_group_id,
            category=category,
        )

    def _log_text(self, tag: str, text: str, category: Optional[str] = None) -> None:
        self._client.logged_data_service_client.log_text_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=text,
            category=category,
        )

    def _log_markdown(
        self, tag: str, text: str, category: Optional[str] = None
    ) -> None:
        self._client.logged_data_service_client.log_markdown_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=text,
            category=category,
        )

    def _log_number(
        self, tag: str, number: Union[float, int], category: Optional[str] = None
    ) -> None:
        self._client.logged_data_service_client.log_numeric_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=str(number),
            category=category,
        )

    def _log_boolean(
        self, tag: str, bool_val: bool, category: Optional[str] = None
    ) -> None:
        self._client.logged_data_service_client.log_boolean_data(
            train_id=self._train_id,
            dataset_build_id=self._dataset_build_id,
            tag=tag,
            data=str(bool_val),
            category=category,
        )

    def _log_dataframe(
        self, tag: str, df: pd.DataFrame, category: Optional[str] = None
    ) -> None:
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
            category=category,
        )

    def _log_file_or_directory_from_path(
        self, tag: str, path: Path, category: Optional[str] = None
    ) -> None:
        if path.is_file():
            self._log_binary_from_path(
                tag,
                path,
                LoggedDataType.LOGGED_DATA_TYPE_FILE,
                max_file_size_mb=100,
                category=category,
            )
        elif path.is_dir():
            with tempfile.NamedTemporaryFile() as target_file:
                shutil.make_archive(target_file.name, "zip", path)

                self._log_binary_from_path(
                    tag,
                    Path(f"{target_file.name}.zip"),
                    LoggedDataType.LOGGED_DATA_TYPE_DIRECTORY,
                    max_file_size_mb=100,
                    category=category,
                )

    def _log_video_from_path(
        self, tag: str, path: Path, category: Optional[str] = None
    ) -> None:
        self._log_binary_from_path(
            tag,
            path,
            LoggedDataType.LOGGED_DATA_TYPE_VIDEO,
            max_file_size_mb=100,
            category=category,
        )

    def _log_image_from_path(
        self,
        tag: str,
        path: Path,
        epoch: Optional[int] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_binary_from_path(
            tag,
            path,
            LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
            max_file_size_mb=1,
            epoch=epoch,
            category=category,
        )

    def _log_binary_from_path(
        self,
        tag: str,
        path: Path,
        logged_data_type: "LoggedDataType.V",
        *,
        max_file_size_mb: int,
        epoch: Optional[int] = None,
        category: Optional[str] = None,
    ) -> None:
        file_size_in_bytes = path.stat().st_size
        self._check_size_less_than_mb(file_size_in_bytes, max_file_size_mb)
        with requests.Session() as s, open(path, "rb") as binary_file:
            presigned_url = self._client.logged_data_service_client.log_binary_data(
                train_id=self._train_id,
                dataset_build_id=self._dataset_build_id,
                tag=tag,
                logged_data_type=logged_data_type,
                epoch=epoch,
                category=category,
            )
            resp = s.put(presigned_url, data=binary_file)
            resp.raise_for_status()

    def _log_pil_image(
        self,
        tag: str,
        image: "PIL.Image.Image",
        *,
        epoch: Optional[int] = None,
        category: Optional[str] = None,
    ) -> None:
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
                logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
                epoch=epoch,
                category=category,
            )
            resp = s.put(presigned_url, data=buffer.getvalue())
            resp.raise_for_status()

    def _log_plot_figure(
        self,
        tag: str,
        figure: "matplotlib.figure.Figure",
        category: Optional[str] = None,
    ) -> None:
        with requests.Session() as s, io.BytesIO() as buffer:
            figure.savefig(buffer, format="jpg")
            self._check_buffer_size(buffer=buffer)
            presigned_url = self._client.logged_data_service_client.log_binary_data(
                train_id=self._train_id,
                dataset_build_id=self._dataset_build_id,
                tag=tag,
                logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
                category=category,
            )
            resp = s.put(presigned_url, data=buffer.getvalue())
            resp.raise_for_status()

    def _log_current_plot_figure(
        self, tag: str, plt: ModuleType, category: Optional[str] = None
    ) -> None:
        if len(plt.get_fignums()) > 0:
            self._log_plot_figure(tag, plt.gcf(), category=category)
        else:
            raise ValueError("No figures in the current pyplot state!")

    @staticmethod
    def _check_epoch(epoch: Any) -> None:
        if epoch is not None:
            if not isinstance(epoch, int) or epoch < 0:
                raise ValueError(
                    f"epoch (i.e. step) can only be a non-negative integer, given value: {epoch}"
                )

    @staticmethod
    def _convert_dict_to_dataframe(dictionary: Dict[str, Any]) -> pd.DataFrame:
        new_values = []
        for value in dictionary.values():
            if isinstance(value, (float, int, str, bool)):
                new_values.append(value)
            else:
                new_values.append(str(value))
        df = pd.DataFrame({"name": dictionary.keys(), "value": new_values})  # type: ignore
        df = df.set_index("name")
        return df

    @staticmethod
    def _is_tabular_dict(maybe_dict: Any) -> bool:
        if not isinstance(maybe_dict, dict):
            return False

        for key in maybe_dict:
            if not isinstance(key, str):
                return False

        return True

    @staticmethod
    def _is_video_from_path(value: Any) -> bool:
        return isinstance(value, Path) and has_allowed_extension(
            value, [".mp4", ".webm", ".ogg"]
        )

    def _is_plot_figure(self, value: Any) -> bool:
        return "matplotlib.figure" in get_base_module_list(value)

    def _is_axes_subplot(self, value: Any) -> bool:
        base_module_list = get_base_module_list(value)
        return "matplotlib.axes._subplots" in base_module_list

    def _is_pyplot(self, value: Any) -> bool:
        return (
            hasattr(value, "__name__")
            and "matplotlib.pyplot" == value.__name__
            and isinstance(value, ModuleType)
        )

    def _check_buffer_size(self, buffer: io.BytesIO) -> None:
        size_in_bytes = buffer.tell()
        self._check_size_less_than_mb(size_in_bytes, 1)

    def _check_size_less_than_mb(
        self, size_in_bytes: float, max_mb_size: float
    ) -> None:
        size_in_mb = size_in_bytes / 1000**2
        if size_in_mb > max_mb_size:
            raise ValueError(
                f"Image size cannot exceed {max_mb_size}MB. Current size: {size_in_mb}"
            )
