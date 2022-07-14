import pytest

from layer.projects.utils import validate_project_name


class TestProjectUtils:
    @pytest.mark.parametrize(
        "invalid_name",
        [
            "project name",
            " project-name",
            "project-name ",
            "Test/ name",
            "&project",
            "#name/@name",
            "wrong\\slash",
            "first/correct_name and then something",
        ],
    )
    def test_throw_on_invalid_name(self, invalid_name: str) -> None:
        with pytest.raises(
            ValueError,
            match="(Invalid project or account name. Please provide *)",
        ):
            validate_project_name(invalid_name)

    @pytest.mark.parametrize(
        "valid_name",
        [
            "project-name",
            "MyAccount/MyProject_",
            "__project_name",
            "my_account/my_project",
        ],
    )
    def test_do_not_throw_on_valid_name(self, valid_name: str) -> None:
        validate_project_name(valid_name)
