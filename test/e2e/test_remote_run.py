from layer.contracts.projects import Project
from layer.decorators.dataset_decorator import dataset
from layer.decorators.model_decorator import model
from layer.executables.runner import remote_run


def test_remote_submit(initialized_project: Project):
    @dataset("d")
    def d():
        return [1]

    @model("m")
    def m():
        return [2]

    run = remote_run([d, m])

    assert run
    assert len(run.id.value) > 0
    assert run.project_full_name == initialized_project.full_name
