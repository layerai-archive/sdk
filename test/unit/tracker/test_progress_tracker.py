from yarl import URL

from layer.contracts.asset import AssetType
from layer.tracker.ui_progress_tracker import UIRunProgressTracker


TEST_URL = URL("https://test.layer.ai")


def test_tracker_mark_model_training():
    tracker = UIRunProgressTracker(
        url=TEST_URL,
    )

    tracker.mark_running(
        asset_type=AssetType.MODEL,
        name="model-a",
        tag="1.2",
    )
