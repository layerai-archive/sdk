# type: ignore
import logging
import platform
from pathlib import Path

import pytest

import layer
from layer.flavors import (
    CatBoostModelFlavor,
    CustomModelFlavor,
    HuggingFaceModelFlavor,
    KerasModelFlavor,
    LightGBMModelFlavor,
    PyTorchModelFlavor,
    ScikitLearnModelFlavor,
    TensorFlowModelFlavor,
    XGBoostModelFlavor,
)
from layer.flavors.utils import get_flavor_for_model


logger = logging.getLogger(__name__)


@pytest.mark.skipif(platform.system() == "Darwin", reason="Segfaults on Mac")
class TestModelFlavors:
    def test_lightgbm_flavor(self):
        import lightgbm as lgb
        import numpy as np

        data = np.random.rand(10, 10)
        label = np.random.randint(2, size=10)  # binary target
        train_data = lgb.Dataset(data, label=label)
        model = lgb.train({}, train_data, 2)

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == LightGBMModelFlavor.__name__

    def test_xgboost_flavor(self):
        import numpy as np
        import xgboost as xgb

        x_train = np.random.random(size=(10, 5))
        y_train = np.random.random(size=(10, 1))
        dtrain = xgb.DMatrix(x_train, label=y_train)
        model = xgb.train({}, dtrain, num_boost_round=2)

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == XGBoostModelFlavor.__name__

    def test_xgboost_regressor_flavor(self):
        import xgboost as xgb

        model = xgb.XGBRegressor()
        flavor = MLModelService.get_model_flavor(model, logger)
        assert type(flavor).__name__ == XGBoostModelFlavor.__name__

    def test_sklearn_flavor(self):
        from sklearn import datasets, svm

        clf = svm.SVC()
        iris = datasets.load_iris()
        model = clf.fit(iris.data, iris.target_names[iris.target])

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == ScikitLearnModelFlavor.__name__

    def test_tensorflow_flavor(self):
        import tensorflow as tf

        class Adder(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def add(self, x):
                return x + x + 1.0

        model = Adder()

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == TensorFlowModelFlavor.__name__

    def test_catboost_classifier_flavor(self):
        from catboost import CatBoostClassifier
        from sklearn import datasets

        model = CatBoostClassifier()
        iris = datasets.load_iris()
        model.fit(iris.data, iris.target_names[iris.target], verbose=False)

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_catboost_regressor_flavor(self):
        from catboost import CatBoostRegressor

        train_data = [[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]]

        train_labels = [10, 20, 30]
        model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
        model.fit(train_data, train_labels)

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_catboost_flavor(self):
        from catboost import CatBoost, Pool

        train_data = [["France", 1924, 44], ["USA", 1932, 37], ["USA", 1980, 37]]

        cat_features = [0]
        train_label = [1, 1, 0]

        train_dataset = Pool(
            data=train_data, label=train_label, cat_features=cat_features
        )

        cb = CatBoost({"iterations": 10})
        model = cb.fit(train_dataset)

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == CatBoostModelFlavor.__name__

    def test_keras_flavor(self):
        import keras
        import tensorflow.python.keras

        keras_model = keras.models.Sequential()
        keras_model.add(keras.layers.Dense(2, activation="relu", input_dim=2))
        flavor = get_flavor_for_model(keras_model)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        keras_model = tensorflow.keras.Sequential()
        keras_model.add(keras.layers.Dense(2, activation="relu", input_dim=2))
        flavor = get_flavor_for_model(keras_model)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        keras_model = tensorflow.keras.Sequential()
        keras_model.add(keras.layers.Dense(2, activation="relu", input_dim=2))
        flavor = get_flavor_for_model(keras_model)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer()
        fit_text = "The earth is an awesome place live"
        tokenizer.fit_on_texts(fit_text)
        flavor = get_flavor_for_model(tokenizer)
        assert type(flavor).__name__ == KerasModelFlavor.__name__

    def test_transformers_package(self):
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            TFBertForSequenceClassification,
        )

        model = TFBertForSequenceClassification(BertConfig())
        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == HuggingFaceModelFlavor.__name__

        model = BertForSequenceClassification(BertConfig())
        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == HuggingFaceModelFlavor.__name__

    def test_transformers_type_detection(self, tmp_path):
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            GPT2Config,
            TFBertForSequenceClassification,
            TFGPT2LMHeadModel,
        )

        hf_flavor = HuggingFaceModelFlavor()

        # Bert Tensorflow Model
        tmp_path = tmp_path / "model1"
        model = TFBertForSequenceClassification(BertConfig())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model = hf_flavor.load_model_from_directory(tmp_path).model_object
        assert isinstance(loaded_model, TFBertForSequenceClassification)

        # Bert Pytorch Model
        tmp_path = tmp_path / "model2"
        model = BertForSequenceClassification(BertConfig())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model = hf_flavor.load_model_from_directory(tmp_path).model_object
        assert isinstance(loaded_model, BertForSequenceClassification)

        # GPT2 LMHEad Tensorflow Model
        tmp_path = tmp_path / "model3"
        model = TFGPT2LMHeadModel(GPT2Config())
        hf_flavor.save_model_to_directory(model, tmp_path)
        loaded_model = hf_flavor.load_model_from_directory(tmp_path).model_object

        assert isinstance(loaded_model, TFGPT2LMHeadModel)

    def test_pytorch_flavor(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == PyTorchModelFlavor.__name__

    def test_pytorch_save_load(self, tmp_path):
        import torch

        pt_flavor = PyTorchModelFlavor()
        pt_model = torch.nn.Sequential()
        pt_flavor.save_model_to_directory(pt_model, tmp_path)
        loaded_model = pt_flavor.load_model_from_directory(tmp_path).model_object
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
        model = pt_flavor.load_model_from_directory(tmp_path).model_object
        assert model.__class__.__name__ == "AutoShape"

    def test_custom_flavor(self, tmp_path):
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        from layer import CustomModel

        class DummyModel(CustomModel):
            def __init__(self):
                super().__init__()
                svc = SVC()
                svc.set_params(kernel="linear")
                x, y = load_iris(return_X_y=True)
                svc.fit(x, y)
                self.model = svc

            def predict(self, model_input):
                return self.model.predict(model_input)

        model = DummyModel()
        flavor = get_flavor_for_model(model)
        assert type(flavor).__name__ == CustomModelFlavor.__name__

        flavor.save_model_to_directory(model, tmp_path)
        loaded_model = flavor.load_model_from_directory(tmp_path).model_object
        assert isinstance(loaded_model, layer.CustomModel)
        assert isinstance(loaded_model.model, SVC)

        x, _ = load_iris(return_X_y=True)
        result = loaded_model.predict(x[:5])
        assert list(result) == [0, 0, 0, 0, 0]
