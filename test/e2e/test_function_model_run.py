from sklearn.svm import SVC

import layer
from layer import context
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
