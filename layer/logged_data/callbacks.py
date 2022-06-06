import importlib.util
from typing import Any, Optional, cast

import pandas as pd
import pkg_resources
from packaging import version

import layer


# We need keras and xgboost libraries to be able to define XGBoostCallback and KerasCallback convenience classes.
# Since these are quite large libraries, instead of directly bundling them into our Python package, we use a hacky
# solution to have a "skeleton" base class if the Python environment doesn't already have these libraries installed.
# Given that one would almost certainly have these libraries installed if they decide to use the convenience classes,
# this is a good tradeoff.
# XGBoost implemented the callback interface at version 1.3.0, make sure we have a version newer than that.
if importlib.util.find_spec("xgboost") is None or version.parse(
    pkg_resources.get_distribution("xgboost").version
) < version.parse("1.3.0"):

    class XGBoostTrainingCallback:
        def before_training(self, model: Any) -> Any:
            pass

        def after_training(self, model: Any) -> Any:
            pass

        def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:  # type: ignore
            pass

else:
    import xgboost as xgb

    class XGBoostTrainingCallback(xgb.callback.TrainingCallback):  # type: ignore
        pass


if importlib.util.find_spec("keras") is None:

    class KerasTrainingCallback:
        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # type: ignore
            pass

else:
    import keras  # type: ignore

    class KerasTrainingCallback(keras.callbacks.Callback):  # type: ignore
        pass


class XGBoostCallback(XGBoostTrainingCallback):
    """
    A default implementation for `callbacks` parameter in various XGBoost methods that uses `layer.log`.

    This class is purely for convenience and one can create their own implementations with different overrides.
    For more information, please consult to XGBoost documentation at https://xgboost.readthedocs.io/en/stable/python/callbacks.html
    """

    def __init__(self, importance_type: str = "gain") -> None:
        super().__init__()
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
        if evals_log:
            layer.log(evals_log, epoch)
        return False


class KerasCallback(KerasTrainingCallback):
    """
    A default implementation for `callbacks` parameter in various Keras methods that uses `layer.log`.

    This class is purely for convenience and one can create their own implementations with different overrides.
    For more information, please consult to Keras documentation at https://keras.io/api/callbacks/
    """

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # type: ignore
        if logs:
            layer.log(logs, epoch)
