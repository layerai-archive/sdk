from typing import Any, Optional, cast

import keras  # type: ignore
import pandas as pd
import xgboost as xgb

import layer


class XGBoostCallback(xgb.callback.TrainingCallback):
    """
    A default implementation for `callbacks` parameter in various XGBoost methods that uses `layer.log`.

    This class is purely for convenience and one can create their own implementations with different overrides.
    For more information, please consult to XGBoost documentation at https://xgboost.readthedocs.io/en/stable/python/callbacks.html
    """

    def __init__(self, importance_type: str = "gain") -> None:
        super().__init__()  # type: ignore
        self.importance_type = importance_type

    def before_training(self, model: Any) -> Any:
        return model

    def after_training(self, model: Any) -> Any:
        if model.attr("best_score") is not None:
            print(
                {
                    "best_score": float(cast(str, model.attr("best_score"))),
                    "best_iteration": int(cast(str, model.attr("best_iteration"))),
                }
            )

        features = model.get_score(importance_type=self.importance_type)
        importance = [[feature, features[feature]] for feature in features]

        df = pd.DataFrame(data=importance, columns=["feature", "importance"])
        layer.log({"Feature Importance Table": df})
        layer.log(
            {"Feature Importance": df.plot.bar(x="feature", y="importance", rot=0)}  # type: ignore
        )

        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:  # type: ignore
        layer.log(evals_log, epoch)
        return False


class KerasCallback(keras.callbacks.Callback):
    """
    A default implementation for `callbacks` parameter in various Keras methods that uses `layer.log`.

    This class is purely for convenience and one can create their own implementations with different overrides.
    For more information, please consult to Keras documentation at https://keras.io/api/callbacks/
    """

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # type: ignore
        layer.log(logs, epoch)  # type: ignore
