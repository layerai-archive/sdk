import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import pytest
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType
from requests import Session

import layer
from layer.clients.layer import LayerClient
from layer.clients.logged_data_service import LoggedDataClient, ModelMetricPoint
from layer.logged_data.log_data_runner import LogDataRunner


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_data_with_string_value_then_calls_log_text_data(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag1 = "string-tag"
    string_value1 = "string-value"
    tag2 = "string-tag"
    string_value2 = "string-value"

    # when
    runner.log({tag1: string_value1, tag2: string_value2})

    # then
    logged_data_client.log_text_data.assert_any_call(
        train_id=train_id,
        tag=tag1,
        data=string_value1,
        dataset_build_id=dataset_build_id,
    )
    logged_data_client.log_text_data.assert_any_call(
        train_id=train_id,
        tag=tag2,
        data=string_value2,
        dataset_build_id=dataset_build_id,
    )


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_data_with_bool_value_then_calls_log_boolean_data(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "bool-tag"
    boolean_value = False

    # when
    runner.log({tag: boolean_value})

    # then
    logged_data_client.log_boolean_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        data=str(boolean_value),
        dataset_build_id=dataset_build_id,
    )


def test_given_runner_when_log_numeric_value_without_epoch_then_calls_log_number() -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    train_id = uuid.uuid4()
    runner = LogDataRunner(client=client, train_id=train_id, logger=None)
    tag1 = "numeric-value-tag-1"
    numeric_value1 = 2.3

    # when
    runner.log({tag1: numeric_value1})

    # then
    logged_data_client.log_numeric_data.assert_any_call(
        train_id=train_id,
        tag=tag1,
        data=str(numeric_value1),
        dataset_build_id=None,
    )


def test_given_runner_when_log_numeric_value_with_epoch_then_calls_log_metric() -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    train_id = uuid.uuid4()
    runner = LogDataRunner(client=client, train_id=train_id, logger=None)
    tag1 = "numeric-value-tag-1"
    numeric_value1 = 2.3
    tag2 = "numeric-value-tag-2"
    numeric_value2 = 2.5
    epoch = 123

    # when
    runner.log({tag1: numeric_value1, tag2: numeric_value2}, epoch=epoch)

    # then
    logged_data_client.log_model_metric.assert_any_call(
        train_id=train_id,
        tag=tag1,
        points=[ModelMetricPoint(epoch=epoch, value=numeric_value1)],
        metric_group_uuid=ANY,
    )

    logged_data_client.log_model_metric.assert_any_call(
        train_id=train_id,
        tag=tag2,
        points=[ModelMetricPoint(epoch=epoch, value=numeric_value2)],
        metric_group_uuid=ANY,
    )


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_dict_then_calls_log_table(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "dict-tag"
    parameters = {
        "tom": 10,
        "nick": 15,
        "juli": 14,
        "jack": [1, 2, 3],
    }

    # when
    runner.log({tag: parameters})

    # then
    expected_dataframe = pd.DataFrame(
        [["tom", 10], ["nick", 15], ["juli", 14], ["jack", "[1, 2, 3]"]],
        columns=["name", "value"],
    )
    expected_dataframe = expected_dataframe.set_index("name")
    expected_dataframe_in_json = expected_dataframe.to_json(orient="table")
    logged_data_client.log_table_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        data=expected_dataframe_in_json,
        dataset_build_id=dataset_build_id,
    )


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_invalid_dict_then_raises(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "dict-tag"

    with pytest.raises(
        ValueError, match=r".*Unsupported value type -> <class 'dict'>.*"
    ):
        runner.log({tag: {10: 20}})


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_pandas_dataframe_then_calls_log_table(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "pandas-dataframe-tag"
    dataframe = pd.DataFrame(
        [["tom", 10], ["nick", 15], ["juli", 14]], columns=["Name", "Age"]
    )
    dataframe_in_json = dataframe.to_json(orient="table")
    # when
    runner.log({tag: dataframe})

    # then
    logged_data_client.log_table_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        data=dataframe_in_json,
        dataset_build_id=dataset_build_id,
    )


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_dataframe_bigger_than_1000_rows_then_raises_error(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "pandas-dataframe-tag"
    dataframe = pd.DataFrame(index=np.arange(1001), columns=np.arange(1))

    # when
    with pytest.raises(ValueError, match=r".*DataFrame rows size cannot exceed 1000.*"):
        runner.log({tag: dataframe})

    # then
    logged_data_client.log_table_data.assert_not_called()


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_nparray_image_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )

    img = np.zeros((100, 100, 3))
    img[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    tag = "nparray-image-tag"
    image = layer.Image(img, format="HWC")

    # when
    runner.log({tag: image})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_hw_nparray_image_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )

    # gradient between 0 and 1 for 256*256
    array = np.linspace(0, 1, 256 * 256)
    # reshape to 2d
    img = np.reshape(array, (256, 256))
    tag = "nparray-image-tag"
    image = layer.Image(img, format="HW")

    # when
    runner.log({tag: image})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_torch_tensor_image_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )

    from torchvision import transforms

    image_data = np.random.rand(400, 400, 3) * 255
    pil_img = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    tensor_image = transforms.ToTensor()(pil_img)

    tag = "torch-tensor-image-tag"
    image = layer.Image(tensor_image, format="HW")

    # when
    runner.log({tag: image})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_image_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "pillow-image-tag"
    image_data = np.random.rand(400, 400, 3) * 255
    image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")

    # when
    runner.log({tag: image})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@patch.object(Session, "put")
def test_given_runner_when_log_image_with_step_for_model_train_then_calls_log_binary(
    mock_put,
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    train_id = uuid.uuid4()
    runner = LogDataRunner(
        client=client,
        train_id=train_id,
        logger=None,
    )
    tag = "pillow-image-tag"
    image_data = np.random.rand(400, 400, 3) * 255
    image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")

    # when
    runner.log({tag: image}, epoch=123)

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        dataset_build_id=None,
        tag=tag,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=123,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_image_bigger_than_1_mb_then_raises_error(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "pillow-big-image-tag"
    image_data = np.random.rand(1000, 1000, 3) * 255
    image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")

    # when
    with pytest.raises(ValueError, match=r".*Image size cannot exceed 1MB.*"):
        runner.log({tag: image})

    # then
    logged_data_client.log_binary_data.assert_not_called()
    mock_put.assert_not_called()


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_matplotlib_figure_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "matplotlib-image-tag"
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

    # when
    runner.log({tag: fig})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_matplotlib_module_then_calls_log_binary(
    mock_put, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # Clean up state for parametrized test
    plt.clf()
    plt.cla()
    plt.close("all")

    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "matplotlib-image-tag"
    # when
    with pytest.raises(
        ValueError, match=r".*No figures in the current pyplot state!.*"
    ):
        runner.log({tag: plt})

    # then
    logged_data_client.log_binary_data.assert_not_called()

    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    unused_fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(
        xlabel="time (s)",
        ylabel="voltage (mV)",
        title="About as simple as it gets, folks",
    )
    ax.grid()

    # when
    runner.log({tag: plt})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_image_by_path_then_calls_log_binary(
    mock_put, tmpdir, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "image-by-path"
    image_data = np.random.rand(100, 100, 3) * 255
    image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    path = tmpdir.join("image.png")
    image.save(str(path))

    # when
    runner.log({tag: Path(str(path))})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
@patch.object(Session, "put")
def test_given_runner_when_log_video_by_path_then_calls_log_binary(
    mock_put, tmpdir, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    logged_data_client.log_binary_data.return_value = "http://path/for/upload"
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "video-by-path"
    path = tmpdir.join("temp.mp4")
    Path(str(path)).touch()

    # when
    runner.log({tag: Path(str(path))})

    # then
    logged_data_client.log_binary_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        dataset_build_id=dataset_build_id,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_VIDEO,
        epoch=None,
    )
    mock_put.assert_called_with("http://path/for/upload", data=ANY)


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_image_by_path_with_unsupported_extension_then_raise(
    tmpdir, train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "image-by-path"
    image_data = np.random.rand(100, 100, 3) * 255
    image = PIL.Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    path = tmpdir.join("image.webp")
    image.save(str(path))

    # when
    with pytest.raises(ValueError, match=r".*Unsupported value type ->.*"):
        runner.log({tag: Path(str(path))})

    # then
    logged_data_client.log_binary_data.assert_not_called()


@pytest.mark.parametrize(
    ("train_id", "dataset_build_id"), [(uuid.uuid4(), None), (None, uuid.uuid4())]
)
def test_given_runner_when_log_markdown_then_calls_log_markdown(
    train_id: Optional[UUID], dataset_build_id: Optional[UUID]
) -> None:
    # given
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )

    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=dataset_build_id
    )
    tag = "markdown-tag"
    md = layer.Markdown("# Foo bar")
    # when
    runner.log({tag: md})

    # then
    logged_data_client.log_markdown_data.assert_called_with(
        train_id=train_id,
        tag=tag,
        data=md.data,
        dataset_build_id=dataset_build_id,
    )


def test_given_runner_with_invalid_epoch_then_raises() -> None:
    logged_data_client = MagicMock(spec=LoggedDataClient)
    client = MagicMock(
        set_spec=LayerClient,
        logged_data_service_client=logged_data_client,
    )
    train_id = uuid.uuid4()
    runner = LogDataRunner(
        client=client, train_id=train_id, logger=None, dataset_build_id=None
    )

    with pytest.raises(
        ValueError, match=r".*can only be a non-negative integer, given value: -1.*"
    ):
        runner.log({"foo": {10: 20}}, epoch=-1)

    with pytest.raises(
        ValueError, match=r".*can only be a non-negative integer, given value: xyz.*"
    ):
        runner.log({"foo": {10: 20}}, epoch="xyz")
