from layer import Context
from layer.context import get_active_context, reset_active_context, set_active_context
from layer.contracts.asset import AssetType


class TestContext:
    def test_correct_context_returned(self) -> None:
        assert get_active_context() is None
        ctx = Context(
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        )
        set_active_context(ctx)
        assert get_active_context() == ctx
        reset_active_context()
        assert get_active_context() is None
