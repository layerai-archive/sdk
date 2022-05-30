import logging
import uuid
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

from layer.clients.layer import LayerClient
from layer.clients.model_catalog import ModelCatalogClient
from layer.config import ClientConfig
from layer.exceptions.exceptions import UnexpectedModelTypeException
from layer.training.train import Train

from .. import IS_DARWIN_ARM64


logger = logging.getLogger(__name__)


def test_log_parameter_parsing() -> None:
    to_return = {
        "param0": "1",
        "param1": "2.4",
        "param2": "asd",
    }
    client = create_autospec(LayerClient)
    client.model_catalog.get_model_train_parameters = MagicMock(return_value=to_return)
    train = Train(
        layer_client=client,
        name="name",
        project_name="test-project",
        version=2,
        train_id=uuid.uuid4(),
    )
    fetched_params = train.get_parameters()
    fetched_param0 = fetched_params["param0"]
    fetched_param1 = fetched_params["param1"]
    fetched_param2 = fetched_params["param2"]
    assert isinstance(fetched_param0, int)
    assert fetched_param0 == 1
    assert isinstance(fetched_param1, float)
    assert fetched_param1 == 2.4
    assert isinstance(fetched_param2, str)
    assert fetched_param2 == "asd"


def test_when_log_parameters_non_string_key_value_then_stringify() -> None:
    parameters = {
        1: "1",
        "param1": 2.4,
    }
    client = create_autospec(LayerClient)
    client.model_catalog.log_parameters = MagicMock()
    train = Train(
        layer_client=client,
        name="name",
        project_name="test-project",
        version=2,
        train_id=uuid.uuid4(),
    )
    train.log_parameters(parameters)
    assert client.model_catalog.log_parameters.called
    for key, value in client.model_catalog.log_parameters.call_args[1][
        "parameters"
    ].items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_when_log_parameter_non_string_key_value_then_stringify() -> None:
    client = create_autospec(LayerClient)
    client.model_catalog.log_parameter = MagicMock()
    train = Train(
        layer_client=client,
        name="name",
        project_name="test-project",
        version=2,
        train_id=uuid.uuid4(),
    )
    train.log_parameter(1, 1)  # type:ignore
    assert client.model_catalog.log_parameter.called
    assert isinstance(client.model_catalog.log_parameter.call_args[1]["name"], str)
    assert isinstance(client.model_catalog.log_parameter.call_args[1]["value"], str)


def test_train_raises_exception_if_error_happens() -> None:
    client = create_autospec(LayerClient)
    client.model_catalog.complete_model_train.side_effect = Exception("cannot complete")
    try:
        with Train(
            layer_client=client,
            name="name",
            project_name="test-project",
            version=2,
            train_id=uuid.uuid4(),
        ):
            raise Exception("train exception")
    except Exception as e:
        assert str(e) == "train exception"


@pytest.mark.parametrize(
    "invalid_model_object",
    [
        "Invalid object type",
        1.23,
        [],
        {},
        set(),
    ],
)
@pytest.mark.skipif(IS_DARWIN_ARM64, reason="Segfaults on Mac M1")
def test_when_save_model_gets_invalid_object_then_throw_exception(
    invalid_model_object: Any,
) -> None:
    config = create_autospec(ClientConfig)
    config.model_catalog = MagicMock()
    config.s3 = MagicMock()
    config.s3.endpoint_url = MagicMock()
    client = create_autospec(LayerClient)
    client.model_catalog = ModelCatalogClient(config, logger)

    train = Train(
        layer_client=client,
        name="name",
        project_name="test-project",
        version=2,
        train_id=uuid.uuid4(),
    )
    with pytest.raises(UnexpectedModelTypeException):
        train.save_model(invalid_model_object)
