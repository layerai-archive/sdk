import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import pytest
from layerapi.api.service.logged_data.logged_data_api_pb2 import LogDataResponse
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType
from requests import Session  # type: ignore

import layer
from layer.clients.logged_data_service import LoggedDataClient
from layer.contracts.logged_data import XCoordinateType
from layer.logged_data.immediate_logged_data_destination import (
    ImmediateLoggedDataDestination,
)
from layer.logged_data.log_data_runner import LogDataRunner


# pylint: disable=too-many-statements
def generate_test_data() -> List[
    Tuple[
        Callable[[Path], Dict[str, Any]],
        Optional[XCoordinateType],
        Callable[[Any], List[Dict[str, Any]]],
    ]
]:
    test_data: List[
        Tuple[
            Callable[[Path], Dict[str, Any]],
            Optional[XCoordinateType],
            Callable[[Any], List[Dict[str, Any]]],
        ]
    ] = []

    # string
    def string_data(tmpdir: Path) -> Dict[str, Any]:
        return {"tag1": "val1", "tag2": "val2"}

    test_data.append(
        (
            # data
            string_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": val,
                    "type": LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                }
            ],
        )
    )

    # number
    def number_data(tmpdir: Path) -> Dict[str, Any]:
        return {"tag1": 1, "tag2": 2.3}

    for x_coordinate_type in list(XCoordinateType):
        test_data.append(
            (
                # data
                number_data,
                # x_coordinate_type
                x_coordinate_type,
                # expected kwargs
                lambda val: [
                    {
                        "value": str(val),
                        "type": LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
                    }
                ],
            )
        )

    # boolean
    def boolean_data(tmpdir: Path) -> Dict[str, Any]:
        return {"tag1": True, "tag2": False}

    test_data.append(
        (
            # data
            boolean_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": str(val),
                    "type": LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
                }
            ],
        )
    )

    # list
    def list_data(tmpdir: Path) -> Dict[str, Any]:
        return {"tag1": [1, 2, "a"], "tag2": ["x", 5.6, ["a", "b"]]}

    test_data.append(
        (
            # data
            list_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": str(val),
                    "type": LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                }
            ],
        )
    )

    # numpy array
    def numpy_data(tmpdir: Path) -> Dict[str, Any]:
        return {"tag1": np.array([1, 2]), "tag2": np.array([[1, 2], [3, 4]])}

    test_data.append(
        (
            # data
            numpy_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": str(val.tolist()),
                    "type": LoggedDataType.LOGGED_DATA_TYPE_TEXT,
                }
            ],
        )
    )

    # dict as grouped logs
    def dict_data(tmpdir: Path) -> Dict[str, Any]:
        return {
            "tag1": {
                "tom": 10,
                "nick": "test",
                "juli": True,
                "jack": [1, 2, 3],
            }
        }

    def dict_get_expected(val: Dict[str, Any]) -> List[Dict[str, Any]]:
        expected = []
        for k, v in val.items():
            logged_data_type = None
            if isinstance(v, bool):
                logged_data_type = LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN
            elif isinstance(v, (float, int)):
                logged_data_type = LoggedDataType.LOGGED_DATA_TYPE_NUMBER
            elif isinstance(v, (str, list)):
                logged_data_type = LoggedDataType.LOGGED_DATA_TYPE_TEXT
            expected.append(
                {
                    "group_tag": "tag1",
                    "tag": k,
                    "value": str(v),
                    "type": logged_data_type,
                }
            )
        return expected

    test_data.append(
        (
            # data
            dict_data,
            # x_coordinate_type
            None,
            # expected kwargs
            dict_get_expected,
        )
    )

    # pandas dataframe
    def pandas_data(tmpdir: Path) -> Dict[str, Any]:
        dataframe = pd.DataFrame(
            [["tom", 10], ["nick", 15], ["juli", 14]], columns=["Name", "Age"]
        )
        return {"tag1": dataframe}

    test_data.append(
        (
            # data
            pandas_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": val.to_json(orient="table"),
                    "type": LoggedDataType.LOGGED_DATA_TYPE_TABLE,
                }
            ],
        )
    )

    #####################
    # images
    #####################
    def expected_image(val: Any) -> List[Dict[str, Any]]:
        return [{"type": LoggedDataType.LOGGED_DATA_TYPE_IMAGE}]

    # PIL image
    def pil_image_data(tmpdir: Path) -> Dict[str, Any]:
        image_data = np.random.rand(400, 400, 3) * 255
        image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        return {"tag1": image}

    for x_coordinate_type in list(XCoordinateType):
        test_data.append(
            (
                # data
                pil_image_data,
                # x_coordinate_type
                x_coordinate_type,
                # expected kwargs
                expected_image,
            )
        )

    # nparray image
    def nparray_image_data(tmpdir: Path) -> Dict[str, Any]:
        img = np.zeros((100, 100, 3))
        img[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
        image = layer.Image(img, format="HWC")
        return {"tag1": image}

    test_data.append(
        (
            # data
            nparray_image_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    # hw nparray image
    def hw_nparray_image_data(tmpdir: Path) -> Dict[str, Any]:
        # gradient between 0 and 1 for 256*256
        nparray = np.linspace(0, 1, 256 * 256)
        # reshape to 2d
        img = np.reshape(nparray, (256, 256))
        image = layer.Image(img, format="HW")
        return {"tag1": image}

    test_data.append(
        (
            # data
            hw_nparray_image_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    # torch tensor image
    def torch_image_data(tmpdir: Path) -> Dict[str, Any]:
        import torch

        img_width = 480
        img_height = 640
        image_tensor = torch.rand((img_height, img_width, 3))
        image = layer.Image(image_tensor, format="HWC")
        image_object = image.get_image()
        assert isinstance(image_object, PIL.Image.Image)
        assert image_object.width == img_width
        assert image_object.height == img_height
        return {"tag1": image}

    test_data.append(
        (
            # data
            torch_image_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    # hw torch tensor image
    def hw_torch_image_data(tmpdir: Path) -> Dict[str, Any]:
        from torchvision import transforms

        image_data = np.random.rand(400, 400, 3) * 255
        pil_img = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        tensor_image = transforms.ToTensor()(pil_img)
        image = layer.Image(tensor_image, format="HW")
        return {"tag1": image}

    test_data.append(
        (
            # data
            hw_torch_image_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    # matplotlib figure
    def matplotlib_figure_data(tmpdir: Path) -> Dict[str, Any]:
        # Clean up state for parametrized test
        plt.clf()
        plt.cla()
        plt.close("all")
        # Data for plotting
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(
            xlabel="time (s)",
            ylabel="voltage (mV)",
            title="About as simple as it gets, folks",
        )
        ax.grid()
        return {"tag1": fig}

    test_data.append(
        (
            # data
            matplotlib_figure_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    # image by path
    def image_path_data(tmpdir: Path) -> Dict[str, Any]:
        image_data = np.random.rand(100, 100, 3) * 255
        image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        path = tmpdir / "image.png"
        image.save(path)
        return {"tag1": path}

    test_data.append(
        (
            # data
            image_path_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_image,
        )
    )

    #####################
    # videos
    #####################
    def expected_video(val: Any) -> List[Dict[str, Any]]:
        return [{"type": LoggedDataType.LOGGED_DATA_TYPE_VIDEO}]

    # torch tensor video
    def torch_tensor_video_data(tmpdir: Path) -> Dict[str, Any]:
        import torch

        video_width = 480
        video_height = 640
        video_tensor = torch.rand((10, 3, video_width, video_height))
        video = layer.Video(video_tensor)
        video_object = video.get_video()
        assert video_object is not None

        return {"tag1": video}

    test_data.append(
        (
            # data
            torch_tensor_video_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_video,
        )
    )

    # video by path
    def video_path_data(tmpdir: Path) -> Dict[str, Any]:
        path = tmpdir / "video.mp4"
        path.touch()
        return {"tag1": path}

    test_data.append(
        (
            # data
            video_path_data,
            # x_coordinate_type
            None,
            # expected kwargs
            expected_video,
        )
    )

    # markdown
    def markdown_data(tmpdir: Path) -> Dict[str, Any]:
        md = layer.Markdown("# Foo bar")
        return {"tag1": md}

    test_data.append(
        (
            # data
            markdown_data,
            # x_coordinate_type
            None,
            # expected kwargs
            lambda val: [
                {
                    "value": val.data,
                    "type": LoggedDataType.LOGGED_DATA_TYPE_MARKDOWN,
                }
            ],
        )
    )

    return test_data


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@pytest.mark.parametrize(("group_tag",), [(None,), ("group",)])
@pytest.mark.parametrize(("category",), [(None,), ("category",)])
@pytest.mark.parametrize(
    ("get_data", "x_coordinate_type", "get_expected_kwargs"),
    generate_test_data(),
)
@patch.object(Session, "put")
def test_log_data(
    mock_put: MagicMock,
    tmp_path: Path,
    train_id: Optional[UUID],
    dataset_build_id: Optional[UUID],
    group_tag: Optional[str],
    category: Optional[str],
    get_data: Callable[[Path], Mapping[str, Any]],
    x_coordinate_type: Optional[XCoordinateType],
    get_expected_kwargs: Callable[[Any], Sequence[Mapping[str, Any]]],
) -> None:
    logged_data_client = MagicMock(spec=LoggedDataClient)
    logged_data_client.log_data.return_value = MagicMock(
        spec=LogDataResponse,
        s3_path="http://path/for/upload",
    )

    runner = LogDataRunner(
        logged_data_destination=ImmediateLoggedDataDestination(logged_data_client),
        dataset_build_id=dataset_build_id,
        train_id=train_id,
        logger=None,
    )

    # given
    data = get_data(tmp_path)
    log_kwargs: Dict[str, Any] = {
        "group_tag": group_tag,
        "category": category,
    }
    if x_coordinate_type and x_coordinate_type != XCoordinateType.INVALID:
        log_kwargs.update(
            {
                "x_coordinate": 123,
                "x_coordinate_type": x_coordinate_type,
            }
        )

    # when
    runner.log(data, **log_kwargs)

    # then
    for tag, value in data.items():
        for test_expected_kwargs in get_expected_kwargs(value):
            expected_kwargs = {
                "dataset_build_id": dataset_build_id,
                "train_id": train_id,
                "tag": tag,
                **log_kwargs,
                **test_expected_kwargs,
            }
            logged_data_client.log_data.assert_any_call(**expected_kwargs)
            # calls either need to set value directly or upload the value
            if "value" not in expected_kwargs:
                mock_put.assert_called_with("http://path/for/upload", data=ANY)


def generate_test_error_data() -> List[
    Tuple[
        Callable[[Path], Dict[str, Any]],
        Any,
        str,
    ]
]:
    test_data: List[
        Tuple[
            Callable[[Path], Dict[str, Any]],
            Any,
            str,
        ]
    ] = []

    # invalid dict
    def invalid_dict(tmpdir: Path) -> Dict[str, Any]:
        return {"data": {"tag1": {10: 20}}}

    test_data.append(
        (
            # kwargs
            invalid_dict,
            # expected error
            ValueError,
            # expected error pattern
            r".*Unsupported value type -> <class 'dict'>.*",
        )
    )

    # more than 1000 rows
    def large_dataframe(tmpdir: Path) -> Dict[str, Any]:
        dataframe = pd.DataFrame(index=np.arange(1001), columns=np.arange(1))
        return {"data": {"tag1": dataframe}}

    test_data.append(
        (
            # kwargs
            large_dataframe,
            # expected error
            ValueError,
            # expected error pattern
            r".*DataFrame rows size cannot exceed 1000.*",
        )
    )

    # larger than 1 mb
    def large_image(tmpdir: Path) -> Dict[str, Any]:
        image_data = np.random.rand(1000, 1000, 3) * 255
        image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        return {"data": {"tag1": image}}

    test_data.append(
        (
            # kwargs
            large_image,
            # expected error
            ValueError,
            # expected error pattern
            r".*Image size cannot exceed 1MB.*",
        )
    )

    # invalid x coordinate
    def invalid_x_coordinate_negative(tmpdir: Path) -> Dict[str, Any]:
        return {"data": {"tag1": 1}, "x_coordinate": -1}

    test_data.append(
        (
            # kwargs
            invalid_x_coordinate_negative,
            # expected error
            ValueError,
            # expected error pattern
            r".*can only be a non-negative integer, given value: -1.*",
        )
    )

    def invalid_x_coordinate_non_integer(tmpdir: Path) -> Dict[str, Any]:
        return {"data": {"tag1": 1}, "x_coordinate": "xyz"}

    test_data.append(
        (
            # kwargs
            invalid_x_coordinate_non_integer,
            # expected error
            ValueError,
            # expected error pattern
            r".*can only be a non-negative integer, given value: xyz.*",
        )
    )

    return test_data


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@pytest.mark.parametrize(
    ("get_kwargs", "expected_error", "expected_error_pattern"),
    generate_test_error_data(),
)
@patch.object(Session, "put")
def test_log_data_raises_error(
    mock_put: MagicMock,
    tmp_path: Path,
    train_id: Optional[UUID],
    dataset_build_id: Optional[UUID],
    get_kwargs: Callable[[Path], Mapping[str, Any]],
    expected_error: Any,
    expected_error_pattern: str,
) -> None:
    logged_data_client = MagicMock()
    logged_data_client.log_data.return_value = MagicMock(
        spec=LogDataResponse,
        s3_path="http://path/for/upload",
    )

    runner = LogDataRunner(
        dataset_build_id=dataset_build_id,
        train_id=train_id,
        logger=None,
        logged_data_destination=ImmediateLoggedDataDestination(logged_data_client),
    )

    # given
    kwargs = get_kwargs(tmp_path)

    # when
    with pytest.raises(expected_error, match=expected_error_pattern):
        runner.log(**kwargs)

    # then
    logged_data_client.log_data.assert_not_called()
    mock_put.assert_not_called()
