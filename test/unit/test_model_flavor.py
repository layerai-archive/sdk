# type: ignore
import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import keras
import lightgbm as lgb
import numpy as np
import tensorflow as tf
import tensorflow.python.keras
import xgboost as xgb
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from keras.layers import Dense
from layerapi.api.ids_pb2 import ModelTrainId
from sklearn import datasets, svm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    TFBertForSequenceClassification,
    TFGPT2LMHeadModel,
)

from layer.clients.model_service import MLModelService
from layer.contracts.runs import ResourceTransferState
from layer.flavors import (
    CatBoostModelFlavor,
    HuggingFaceModelFlavor,
    KerasModelFlavor,
    LightGBMModelFlavor,
    PyTorchModelFlavor,
    ScikitLearnModelFlavor,
    TensorFlowModelFlavor,
    XGBoostModelFlavor,
)


logger = logging.getLogger(__name__)


class TestModelFlavors:
    def test_lightgbm_flavor(self):
        data = np.random.rand(10, 10)
        label = np.random.randint(2, size=10)  # binary target
        train_data = lgb.Dataset(data, label=label)
        model = lgb.train({}, train_data, 2)

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == LightGBMModelFlavor.__name__

    def test_xgboost_flavor(self):
        x_train = np.random.random(size=(10, 5))
        y_train = np.random.random(size=(10, 1))
        dtrain = xgb.DMatrix(x_train, label=y_train)
        model = xgb.train({}, dtrain, num_boost_round=2)

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == XGBoostModelFlavor.__name__

    def test_sklearn_flavor(self):
        clf = svm.SVC()
        iris = datasets.load_iris()
        model = clf.fit(iris.data, iris.target_names[iris.target])

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == ScikitLearnModelFlavor.__name__

    def test_tensorflow_flavor(self):
        class Adder(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def add(self, x):
                return x + x + 1.0

        model = Adder()

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == TensorFlowModelFlavor.__name__

    def test_catboost_classifier_flavor(self):
        model = CatBoostClassifier()
        iris = datasets.load_iris()
        model.fit(iris.data, iris.target_names[iris.target], verbose=False)

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_catboost_regressor_flavor(self):
        train_data = [[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]]

        train_labels = [10, 20, 30]
        model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
        model.fit(train_data, train_labels)

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_catboost_flavor(self):
        train_data = [["France", 1924, 44], ["USA", 1932, 37], ["USA", 1980, 37]]

        cat_features = [0]
        train_label = [1, 1, 0]

        train_dataset = Pool(
            data=train_data, label=train_label, cat_features=cat_features
        )

        cb = CatBoost({"iterations": 10})
        model = cb.fit(train_dataset)

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_keras_flavor(self):
        keras_model = keras.models.Sequential()
        keras_model.add(Dense(2, activation="relu", input_dim=2))
        flavor = MLModelService.get_model_flavor(keras_model, logger)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        keras_model = tensorflow.keras.Sequential()
        keras_model.add(Dense(2, activation="relu", input_dim=2))
        flavor = MLModelService.get_model_flavor(keras_model, logger)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        keras_model = tensorflow.keras.Sequential()
        keras_model.add(Dense(2, activation="relu", input_dim=2))
        flavor = MLModelService.get_model_flavor(keras_model, logger)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer()
        fit_text = "The earth is an awesome place live"
        tokenizer.fit_on_texts(fit_text)
        flavor = MLModelService.get_model_flavor(tokenizer, logger)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

    @patch("layer.flavors.base.S3Util", autospec=True)
    def test_keras_save_load(self, mock_s3_util, tmp_path: Path):
        # Move the pickled model to tmp_path of pytest
        def mock_upload_dir(local_dir: Path, credentials, s3_path, endpoint_url, state):
            import shutil

            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            shutil.copytree(local_dir, tmp_path)

        # Move the copied pickled model from tmp_path to local_dir
        def mock_download_dir(
            local_dir: Path, credentials, s3_path, endpoint_url, state
        ):
            import shutil

            if local_dir.exists():
                shutil.rmtree(local_dir)
            shutil.copytree(tmp_path, local_dir)
            shutil.rmtree(tmp_path)

        # Mock S3Util for uploading and downloading pickled models
        mock_s3_util.upload_dir = mock_upload_dir
        mock_s3_util.download_dir = mock_download_dir

        # Init flavor
        keras_flavor = KerasModelFlavor()

        model_definition = MagicMock()
        model_definition.model_train_id = ModelTrainId(value=str(uuid.uuid4()))
        keras_model = tensorflow.keras.Sequential()
        keras_model.add(Dense(2, activation="relu", input_dim=2))
        keras_flavor.save_to_s3(model_definition, keras_model)
        loaded_model, _ = keras_flavor.load_from_s3(
            model_definition, state=ResourceTransferState()
        )
        assert isinstance(loaded_model, tensorflow.keras.Sequential)

        model_definition = MagicMock()
        model_definition.model_train_id = ModelTrainId(value=str(uuid.uuid4()))
        keras_model = keras.models.Sequential()
        keras_model.add(Dense(2, activation="relu", input_dim=2))
        keras_flavor.save_to_s3(model_definition, keras_model)
        loaded_model, _ = keras_flavor.load_from_s3(
            model_definition, state=ResourceTransferState()
        )
        assert isinstance(loaded_model, keras.models.Sequential)

        from keras.preprocessing.text import Tokenizer

        model_definition = MagicMock()
        model_definition.model_train_id = ModelTrainId(value=str(uuid.uuid4()))
        tokenizer = Tokenizer()
        fit_text = "The earth is an awesome place live"
        tokenizer.fit_on_texts(fit_text)
        keras_flavor.save_to_s3(model_definition, tokenizer)
        loaded_tokenizer, _ = keras_flavor.load_from_s3(
            model_definition, state=ResourceTransferState()
        )
        assert isinstance(loaded_tokenizer, keras.preprocessing.text.Tokenizer)

    def test_transformers_package(self):
        model = TFBertForSequenceClassification(BertConfig())
        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == HuggingFaceModelFlavor.__name__

        model = BertForSequenceClassification(BertConfig())
        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == HuggingFaceModelFlavor.__name__

    def test_transformers_type_detection(self, tmp_path):
        hf_flavor = HuggingFaceModelFlavor()

        # Bert Tensorflow Model
        tmp_path = tmp_path / "model1"
        model = TFBertForSequenceClassification(BertConfig())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model, _ = hf_flavor.load_model_from_directory(tmp_path)
        assert isinstance(loaded_model, TFBertForSequenceClassification)

        # Bert Pytorch Model
        tmp_path = tmp_path / "model2"
        model = BertForSequenceClassification(BertConfig())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model, _ = hf_flavor.load_model_from_directory(tmp_path)
        assert isinstance(loaded_model, BertForSequenceClassification)

        # GPT2 LMHEad Tensorflow Model
        tmp_path = tmp_path / "model3"
        model = TFGPT2LMHeadModel(GPT2Config())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model, _ = hf_flavor.load_model_from_directory(tmp_path)

        assert isinstance(loaded_model, TFGPT2LMHeadModel)

    def test_pytorch_flavor(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == PyTorchModelFlavor.__name__

    def test_pytorch_save_load(self, tmp_path):
        import torch

        pt_flavor = PyTorchModelFlavor()
        pt_model = torch.nn.Sequential()
        pt_flavor.save_model_to_directory(pt_model, tmp_path)
        loaded_model, _ = pt_flavor.load_model_from_directory(tmp_path)
        assert isinstance(loaded_model, torch.nn.Module)

    def test_yolo_save_load(self, tmp_path):
        import torch
        from yolov5.models.yolo import AutoShape

        # Load a YOLO model with random weights from torch hub
        model_def = "yolov5s.yaml"
        model = AutoShape(torch.nn.Sequential())

        # Save model
        pt_flavor = PyTorchModelFlavor()
        pt_flavor.save_model_to_directory(model, tmp_path)

        # Check if we have the necessary files to load the model
        p = Path(str(tmp_path)).rglob("*.*")
        files = [x.name for x in p if x.is_file()]
        assert model_def in files

        # Unload the `models` module (installed with torch.hub.load above) to make sure Layer can load the module
        # without YOLO installation
        import sys

        if "models" in sys.modules:
            del sys.modules["models"]
            del sys.modules["utils"]
        if "yolov5" in sys.modules:
            del sys.modules["yolov5"]

        # Load and check the type of the model
        model, _ = pt_flavor.load_model_from_directory(tmp_path)
        assert model.__class__.__name__ == "AutoShape"
