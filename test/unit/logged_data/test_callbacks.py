from unittest.mock import ANY, call, patch

import pytest

import layer

from ... import IS_DARWIN


@pytest.mark.skipif(IS_DARWIN, reason="Segfaults on Mac")
@patch.object(layer, "log")
def test_xgboost(mock_log):
    def train():
        import numpy as np
        import xgboost as xgb

        x_train = np.random.random(size=(10, 5))
        y_train = np.random.random(size=(10, 1))
        dtrain = xgb.DMatrix(x_train, label=y_train)
        xgb.train({}, dtrain, num_boost_round=2, callbacks=[layer.XGBoostCallback()])

    train()

    mock_log.assert_called()
    mock_log.assert_has_calls(
        [
            # after_iteration is actually called twice, but since it has no data to log, we don't have calls to layer.log.
            # Leaving these commented here for future reference.
            # call({}, 0),  # after_iteration first iteration
            # call({}, 1),  # after_iteration second iteration
            call({"Feature Importance Table": ANY}),  # after_training
            call({"Feature Importance": ANY}),  # after_training
        ]
    )


@patch.object(layer, "log")
def test_keras(mock_log):
    def train():
        import pandas as pd
        from keras.layers import Dense
        from keras.models import Sequential

        data = [[11, 143, 94, 33, 146, 36.6, 0.254, 51, 1]]

        dataset = pd.DataFrame(
            data,
            columns=[
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col6",
                "col7",
                "col8",
                "col9",
            ],
        )
        # split into input (X) and output (y) variables
        x = dataset.iloc[:, :8]
        y = dataset.iloc[:, 8]
        # define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation="relu"))
        # compile the keras model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        # fit the keras model on the dataset
        model.fit(x, y, epochs=2, batch_size=10, callbacks=[layer.KerasCallback()])

    train()
    mock_log.assert_called()
    mock_log.assert_has_calls(
        [
            call({"loss": ANY, "accuracy": ANY}, 0),  # on_epoch_end first iteration
            call({"loss": ANY, "accuracy": ANY}, 1),  # on_epoch_end second iteration
        ]
    )
