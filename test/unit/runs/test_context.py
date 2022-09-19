import uuid

from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.runs import context


class TestRunContext:
    def test_last_project_name_returned(self) -> None:
        context.reset_to(
            project_full_name=ProjectFullName("acc", "test"),
            project_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            labels=set(),
        )
        context.reset_to(
            project_full_name=ProjectFullName("acc", "anotherTest"),
            project_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            labels=set(),
        )
        assert context.get_project_full_name().project_name == "anotherTest"

    def test_reset(self) -> None:
        context.set_default_fabric(Fabric.F_SMALL)
        context.set_pip_requirements_file("/path/to/requirements2.txt")
        context.set_pip_packages(["numpy=1.22.2"])
        context.reset_to(
            project_full_name=ProjectFullName("acc", "second-test"),
            project_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            labels=set(),
        )
        assert context.get_account_name() == "acc"
        assert context.get_project_full_name().project_name == "second-test"
        assert context.default_fabric() is None
        assert context.get_pip_packages() is None
        assert context.get_pip_requirements_file() is None

    def test_reset_with_the_same_project_and_account_names_does_reset(self) -> None:
        # given
        context.set_default_fabric(Fabric.F_SMALL)
        context.set_pip_requirements_file("/path/to/requirements2.txt")
        context.set_pip_packages(["numpy=1.22.2"])

        # when
        context.reset_to(
            project_full_name=ProjectFullName("test-acc", "test"),
            project_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            labels={"label-1", "label-2"},
        )

        # then
        assert context.get_project_full_name().project_name == "test"
        assert context.get_labels() == {"label-1", "label-2"}
        assert context.default_fabric() is None
        assert context.get_pip_packages() is None
        assert context.get_pip_requirements_file() is None

    def test_last_fabric_returned(self) -> None:
        assert context.default_fabric() is None
        context.set_default_fabric(Fabric.F_SMALL)
        context.set_default_fabric(Fabric.F_MEDIUM)
        assert context.default_fabric() == Fabric.F_MEDIUM

    def test_pip_requirements_file_returned(self) -> None:
        assert context.get_pip_requirements_file() is None
        context.set_pip_requirements_file("/path/to/requirements.txt")
        context.set_pip_requirements_file("/path/to/requirements2.txt")
        assert context.get_pip_requirements_file() == "/path/to/requirements2.txt"

    def test_pip_packages_returned(self) -> None:
        assert context.get_pip_packages() is None
        context.set_pip_packages(["numpy=1.22.1"])
        context.set_pip_packages(["numpy=1.22.2"])
        pip_packages = context.get_pip_packages()
        assert pip_packages is not None
        assert len(pip_packages) == 1
        assert pip_packages[0] == "numpy=1.22.2"
