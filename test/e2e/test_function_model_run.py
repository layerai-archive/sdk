from sklearn.svm import SVC

import layer
from layer import global_context
from layer.contracts.projects import Project
from layer.decorators import model, pip_requirements
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

    @model(model_name)
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    def train_model():
        from sklearn import datasets

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")
        return result

    # when
    train_model()

    assert global_context.get_active_context() is None

    # then
    mdl = layer.get_model(model_name)
    assert isinstance(mdl.get_train(), SVC)
