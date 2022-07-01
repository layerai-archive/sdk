import os
from typing import Any

from .base_progress_tracker import BaseRunProgressTracker
from .fake_progress_tracker import FakeRunProgressTracker
from .progress_tracker import RunProgressTracker


def get_progress_tracker(*args: Any, **kwargs: Any) -> BaseRunProgressTracker:
    disable_tracker = "LAYER_DISABLE_TRACKING_UI" in os.environ
    if disable_tracker:
        return FakeRunProgressTracker()
    else:
        return RunProgressTracker(*args, **kwargs)
