from layer.contracts.fabrics import Fabric
from layer.runs.context import (
    current_account_name,
    current_label_names,
    current_project_full_name,
    default_fabric,
    get_pip_packages,
    get_pip_requirements_file,
    reset_to,
    set_current_project_full_name,
    set_default_fabric,
    set_pip_packages,
    set_pip_requirements_file,
)


class TestGlobalContext:
    def test_last_project_name_returned(self) -> None:
        set_current_project_full_name("acc/test")
        set_current_project_full_name("acc/anotherTest")
        assert current_project_full_name().project_name == "anotherTest"

    def test_reset(self) -> None:
        set_current_project_full_name("acc/test")
        set_default_fabric(Fabric.F_SMALL)
        set_pip_requirements_file("/path/to/requirements2.txt")
        set_pip_packages(["numpy=1.22.2"])
        reset_to("acc/second-test")
        assert current_account_name() == "acc"
        assert current_project_full_name().project_name == "second-test"
        assert default_fabric() is None
        assert get_pip_packages() is None
        assert get_pip_requirements_file() is None

    def test_reset_with_the_same_project_and_account_names_does_reset(self) -> None:
        # given
        set_current_project_full_name("test-acc/test")
        set_default_fabric(Fabric.F_SMALL)
        set_pip_requirements_file("/path/to/requirements2.txt")
        set_pip_packages(["numpy=1.22.2"])

        # when
        reset_to("test-acc/test", label_names={"label-1", "label-2"})

        # then
        assert current_project_full_name().project_name == "test"
        assert current_label_names() == {"label-1", "label-2"}
        assert default_fabric() is None
        assert get_pip_packages() is None
        assert get_pip_requirements_file() is None

    def test_last_fabric_returned(self) -> None:
        assert default_fabric() is None
        set_default_fabric(Fabric.F_SMALL)
        set_default_fabric(Fabric.F_MEDIUM)
        assert default_fabric() == Fabric.F_MEDIUM

    def test_pip_requirements_file_returned(self) -> None:
        assert get_pip_requirements_file() is None
        set_pip_requirements_file("/path/to/requirements.txt")
        set_pip_requirements_file("/path/to/requirements2.txt")
        assert get_pip_requirements_file() == "/path/to/requirements2.txt"

    def test_pip_packages_returned(self) -> None:
        assert get_pip_packages() is None
        set_pip_packages(["numpy=1.22.1"])
        set_pip_packages(["numpy=1.22.2"])
        pip_packages = get_pip_packages()
        assert pip_packages is not None
        assert len(pip_packages) == 1
        assert pip_packages[0] == "numpy=1.22.2"
