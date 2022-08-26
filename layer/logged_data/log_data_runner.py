import io
import shutil
import tempfile
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType

from layer.contracts.logged_data import (
    Image,
    LogDataType,
    LoggedData,
    Markdown,
    Video,
    XCoordinateType,
)

from .logged_data_destination import LoggedDataDestination
from .utils import get_base_module_list, has_allowed_extension


if TYPE_CHECKING:
    import matplotlib.figure  # type: ignore
    import PIL


class LogDataRunner:
    def __init__(
        self,
        logged_data_destination: LoggedDataDestination,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        logger: Optional[Logger] = None,
    ):
        assert bool(train_id) ^ bool(dataset_build_id)
        self._train_id = train_id
        self._dataset_build_id = dataset_build_id
        self._logger = logger
        self._logged_data_destination = logged_data_destination

    def get_logged_data(self, tag: str) -> LoggedData:
        return self._logged_data_destination.get_logged_data(
            tag=tag, train_id=self._train_id, dataset_build_id=self._dataset_build_id
        )

    def log(  # pylint: disable=too-many-statements
        self,
        data: LogDataType,
        x_coordinate: Optional[int] = None,
        x_coordinate_type: XCoordinateType = XCoordinateType.STEP,
        category: Optional[str] = None,
        group_tag: Optional[str] = None,
    ) -> None:
        LogDataRunner._check_x_coordinate(x_coordinate)
        for tag, value in data.items():
            kwargs: Dict[str, Any] = {
                "train_id": self._train_id,
                "dataset_build_id": self._dataset_build_id,
                "tag": tag,
                "group_tag": group_tag,
                "category": category,
            }
            if isinstance(value, str):
                self._log_simple_data(
                    value=value,
                    type=LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                    **kwargs,
                )
            elif isinstance(value, list):
                self._log_simple_data(
                    value=str(value),
                    type=LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                    **kwargs,
                )
            elif isinstance(value, np.ndarray):
                self._log_simple_data(
                    value=str(value.tolist()),
                    type=LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                    **kwargs,
                )
            # boolean check must be done before numeric check as it also returns true for booleans.
            elif isinstance(value, bool):
                self._log_simple_data(
                    value=str(value),
                    type=LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
                    **kwargs,
                )
            elif isinstance(value, (int, float)):
                if x_coordinate is not None:
                    kwargs.update(
                        {
                            "x_coordinate": x_coordinate,
                            "x_coordinate_type": x_coordinate_type,
                        }
                    )
                self._log_simple_data(
                    value=str(value),
                    type=LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
                    **kwargs,
                )
            elif isinstance(value, Markdown):
                self._log_simple_data(
                    value=value.data,
                    type=LoggedDataType.LOGGED_DATA_TYPE_MARKDOWN,
                    **kwargs,
                )
            elif isinstance(value, pd.DataFrame):
                self._log_dataframe(value=value, **kwargs)
            elif LogDataRunner._is_tabular_dict(value):
                if TYPE_CHECKING:
                    assert isinstance(value, dict)
                dataframe = LogDataRunner._convert_dict_to_dataframe(value)
                self._log_dataframe(value=dataframe, **kwargs)
            elif LogDataRunner._is_video_from_path(value):
                if TYPE_CHECKING:
                    assert isinstance(value, Path)
                self._log_video_from_path(path=value, **kwargs)
            elif isinstance(value, Video):
                video_path = value.get_video()
                self._log_video_from_path(path=video_path, **kwargs)
            elif Image.is_image(value):
                if x_coordinate is not None:
                    kwargs.update(
                        {
                            "x_coordinate": x_coordinate,
                            "x_coordinate_type": x_coordinate_type,
                        }
                    )
                if TYPE_CHECKING:
                    import PIL

                    assert isinstance(value, (Path, PIL.Image.Image, Image))
                self._log_image(image=value, **kwargs)
            elif self._is_plot_figure(value):
                if TYPE_CHECKING:
                    import matplotlib.figure

                    assert isinstance(value, matplotlib.figure.Figure)
                self._log_plot_figure(figure=value, **kwargs)
            elif self._is_axes_subplot(value):
                if TYPE_CHECKING:
                    import matplotlib.axes._subplots  # type: ignore

                    assert isinstance(
                        value,
                        matplotlib.axes._subplots.AxesSubplot,  # pylint: disable=protected-access
                    )
                self._log_plot_figure(figure=value.get_figure(), **kwargs)
            elif self._is_pyplot(value):
                assert isinstance(value, ModuleType)
                self._log_current_plot_figure(plt=value, **kwargs)
            elif isinstance(value, Path):  # This must be below image and video!
                self._log_file_or_directory_from_path(path=value, **kwargs)
            else:
                raise ValueError(f"Unsupported value type -> {type(value)}")

    def _log_simple_data(
        self, value: str, type: "LoggedDataType.V", **kwargs: Any
    ) -> None:
        self._logged_data_destination.receive(
            func=lambda client: client.log_data(
                value=value,
                type=type,
                **kwargs,
            )
        )

    def _log_binary(self, value: Any, type: "LoggedDataType.V", **kwargs: Any) -> None:
        self._logged_data_destination.receive(
            func=lambda client: client.log_data(type=type, **kwargs), data=value
        )

    def _log_binary_from_path(
        self, path: Path, type: "LoggedDataType.V", max_file_size_mb: int, **kwargs: Any
    ) -> None:
        file_size_in_bytes = path.stat().st_size
        self._check_size_less_than_mb(file_size_in_bytes, max_file_size_mb)
        with open(path, "rb") as binary_file:
            self._log_binary(value=binary_file, type=type, **kwargs)

    def _log_dataframe(self, value: pd.DataFrame, **kwargs: Any) -> None:
        rows = len(value.index)
        if rows > 1000:
            raise ValueError(
                f"DataFrame rows size cannot exceed 1000. Current size: {rows}"
            )
        df_json = value.to_json(orient="table")  # type: ignore
        self._log_simple_data(
            value=df_json,
            type=LoggedDataType.LOGGED_DATA_TYPE_TABLE,
            **kwargs,
        )

    def _log_image(
        self, image: Union["PIL.Image.Image", Path, Image], **kwargs: Any
    ) -> None:
        # Users can log images directly with `layer.log({'img':img})` for simple images or with
        # `layer.log({'img':layer.Image(img)})` for advanced image formats like Tensors
        if isinstance(image, Image):
            img = image.get_image()
        else:
            if TYPE_CHECKING:
                import PIL

                assert isinstance(image, Path) or isinstance(image, PIL.Image.Image)
            img = image

        if Image.is_image_path(img):
            if TYPE_CHECKING:
                assert isinstance(img, Path)
            self._log_binary_from_path(
                img,
                LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
                max_file_size_mb=1,
                **kwargs,
            )
        elif Image.is_pil_image(img):
            if TYPE_CHECKING:
                import PIL

                assert isinstance(img, PIL.Image.Image)
            self._log_pil_image(image=img, **kwargs)

    def _log_pil_image(self, image: "PIL.Image.Image", **kwargs: Any) -> None:
        with io.BytesIO() as buffer:
            if image.mode in ["RGBA", "P"]:
                image.save(buffer, format="PNG")
            else:
                image.save(buffer, format="JPEG")
            self._check_buffer_size(buffer=buffer)
            self._log_binary(
                value=buffer.getvalue(),
                type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
                **kwargs,
            )

    def _log_plot_figure(
        self, figure: "matplotlib.figure.Figure", **kwargs: Any
    ) -> None:
        with io.BytesIO() as buffer:
            figure.savefig(buffer, format="jpg")
            self._check_buffer_size(buffer=buffer)
            self._log_binary(
                value=buffer.getvalue(),
                type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
                **kwargs,
            )

    def _log_current_plot_figure(self, plt: ModuleType, **kwargs: Any) -> None:
        if len(plt.get_fignums()) > 0:
            self._log_plot_figure(plt.gcf(), **kwargs)
        else:
            raise ValueError("No figures in the current pyplot state!")

    def _log_file_or_directory_from_path(self, path: Path, **kwargs: Any) -> None:
        if path.is_file():
            self._log_binary_from_path(
                path,
                LoggedDataType.LOGGED_DATA_TYPE_FILE,
                max_file_size_mb=100,
                **kwargs,
            )
        elif path.is_dir():
            with tempfile.NamedTemporaryFile() as target_file:
                archive = shutil.make_archive(target_file.name, "zip", path)
                self._log_binary_from_path(
                    Path(archive),
                    LoggedDataType.LOGGED_DATA_TYPE_DIRECTORY,
                    max_file_size_mb=100,
                    **kwargs,
                )

    def _log_video_from_path(self, path: Path, **kwargs: Any) -> None:
        self._log_binary_from_path(
            path,
            LoggedDataType.LOGGED_DATA_TYPE_VIDEO,
            max_file_size_mb=100,
            **kwargs,
        )

    @staticmethod
    def _check_x_coordinate(x_coordinate: Any) -> None:
        if x_coordinate is not None:
            if not isinstance(x_coordinate, int) or x_coordinate < 0:
                raise ValueError(
                    f"x_coordinate can only be a non-negative integer, given value: {x_coordinate}"
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

    @staticmethod
    def _is_plot_figure(value: Any) -> bool:
        return "matplotlib.figure" in get_base_module_list(value)

    @staticmethod
    def _is_axes_subplot(value: Any) -> bool:
        base_module_list = get_base_module_list(value)
        return "matplotlib.axes._subplots" in base_module_list

    @staticmethod
    def _is_pyplot(value: Any) -> bool:
        return (
            hasattr(value, "__name__")
            and "matplotlib.pyplot" == value.__name__
            and isinstance(value, ModuleType)
        )

    @staticmethod
    def _check_buffer_size(buffer: io.BytesIO) -> None:
        size_in_bytes = buffer.tell()
        LogDataRunner._check_size_less_than_mb(size_in_bytes, 1)

    @staticmethod
    def _check_size_less_than_mb(size_in_bytes: float, max_mb_size: float) -> None:
        size_in_mb = size_in_bytes / 1000**2
        if size_in_mb > max_mb_size:
            raise ValueError(
                f"Image size cannot exceed {max_mb_size}MB. Current size: {size_in_mb}"
            )
