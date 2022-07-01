import os
from typing import Any

from .fake_progress_tracker import FakeRunProgressTracker
from .progress_tracker import RunProgressTracker
from .ui_progress_tracker import UIRunProgressTracker


def get_progress_tracker(*args: Any, **kwargs: Any) -> RunProgressTracker:
    disable_tracker = "LAYER_DISABLE_TRACKING_UI" in os.environ
    if disable_tracker:
        return FakeRunProgressTracker()
    else:
        return UIRunProgressTracker(*args, **kwargs)
