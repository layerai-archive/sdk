from typing import Callable, Optional
from uuid import UUID

from sklearn.svm import SVC

import layer
from layer import Context, context
from layer.clients.layer import LayerClient
from layer.contracts.asset import AssetType
from layer.contracts.projects import Project
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.common_scenarios import (
    remote_run_with_model_train_succeeds_and_registers_metadata,
)


def test_remote_run_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    remote_run_with_model_train_succeeds_and_registers_metadata(asserter)


def test_local_run_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    model_name = "zoo-model"

    @layer.model(model_name)
    @layer.pip_requirements(packages=["scikit-learn==0.23.2"])
    def train_model():
        from sklearn import datasets
        from sklearn.svm import SVC

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")

        ctx = context.get_active_context()
        assert ctx is not None
        assert ctx.asset_type() == AssetType.MODEL
        assert ctx.asset_name() == model_name
        return result

    # when
    train_model()

    assert context.get_active_context() is None

    # then
    mdl = layer.get_model(model_name)
    assert isinstance(mdl.get_train(), SVC)


def test_local_run_with_args_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    model_name = "zoo-model"

    @layer.model(model_name)
    @layer.pip_requirements(packages=["scikit-learn==0.23.2"])
    def train_model(test_arg):
        from sklearn import datasets

        if test_arg != "test_arg":
            return

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")
        layer.save_model(result)

    # when
    train_model("test_arg")

    assert context.get_active_context() is None

    # then
    mdl = layer.get_model(model_name)
    assert isinstance(mdl.get_train(), SVC)


def test_local_run_without_decorator_succeeds_and_registers_metadata(
    initialized_project: Project, client: LayerClient
):
    # given
    model_name = "zoo-model"

    def run_experiment(test_arg, cb: Callable[[Context], None]):
        assert test_arg == "test_arg"

        ctx = context.get_active_context()
        assert ctx is not None
        assert ctx.asset_type() == AssetType.MODEL
        assert ctx.asset_name() == model_name
        cb(ctx)

        layer.log({"test_key": 1.5})

    # and
    class ContextHolder:
        """
        We need to have access to the context after it is gone so we can run assertions
        """

        context: Optional[Context]

        def save(self, ctx: Context):
            self.context = ctx

        def train_id(self) -> UUID:
            assert self.context is not None
            assert self.context.model_train() is not None
            return self.context.model_train().id

    ctx_holder = ContextHolder()

    # when
    layer.model(model_name)(run_experiment)("test_arg", ctx_holder.save)

    # then
    assert context.get_active_context() is None
    # and
    logged_data = client.logged_data_service_client.get_logged_data(
        tag="test_key", train_id=ctx_holder.train_id()
    )
    assert logged_data.value == "1.5"
