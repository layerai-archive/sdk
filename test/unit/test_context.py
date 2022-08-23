from layer import Context
from layer.context import get_active_context
from layer.contracts.asset import AssetType


class ExampleContextHolder:
    def __init__(self, context: Context):
        self._context = context

    def context(self) -> Context:
        return self._context


class TestContext:
    def test_correct_context_returned(self) -> None:
        assert get_active_context() is None
        with Context(
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        ) as ctx:
            assert get_active_context() == ctx

        assert get_active_context() is None

    def test_context_reference_works_after_context_exits(self) -> None:
        assert get_active_context() is None
        with Context(
            asset_name="the-model",
            asset_type=AssetType.MODEL,
        ) as ctx:
            holder = ExampleContextHolder(context=ctx)
            assert get_active_context() == ctx

        assert holder.context() is not None
        assert get_active_context() is None
