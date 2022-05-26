from sklearn.svm import SVC

import layer
from layer import global_context
from layer.contracts.projects import Project
from layer.decorators import model, pip_requirements
from test.e2e.assertion_utils import E2ETestAsserter


def test_remote_run_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    model_name = "foo-model"

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
    run = layer.run([train_model])

    # then
    asserter.assert_run_succeeded(run.run_id)
    mdl = layer.get_model(model_name)
    assert isinstance(mdl.get_train(), SVC)


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
