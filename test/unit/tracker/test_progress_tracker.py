from yarl import URL

from layer.contracts.asset import AssetType
from layer.contracts.project_full_name import ProjectFullName
from layer.tracker.ui_progress_tracker import UIRunProgressTracker


PROJECT = ProjectFullName(
    project_name="proj",
    account_name="acc",
)


TEST_URL = URL("https://test.layer.ai")


def test_tracker_mark_model_training():
    tracker = UIRunProgressTracker(
        project_name=PROJECT.project_name,
        account_name=PROJECT.account_name,
        url=TEST_URL,
    )

    tracker.mark_running(
        asset_type=AssetType.MODEL,
        name="model-a",
        tag="1.2",
    )
