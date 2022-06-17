import uuid
from typing import List, Optional, Sequence

import pytest

from layer.contracts.asset import AssetType
from layer.contracts.assets import AssetPath
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.decorators.definitions import FunctionDefinition
from layer.exceptions.exceptions import ProjectCircularDependenciesException
from layer.projects.execution_planner import (
    _build_graph,
    _find_cycles,
    build_execution_plan,
    check_asset_dependencies,
    drop_independent_entities,
)


TEST_PROJECT_FULL_NAME = ProjectFullName(project_name="test", account_name="test-acc")


class TestProjectExecutionPlanner:
    def test_check_asset_external_dependency(self) -> None:
        ds1 = self._create_mock_dataset("ds1", project_name="another-project-x")
        m1 = self._create_mock_model("m1", ["another-project-x/datasets/ds1"])

        check_asset_dependencies([m1, ds1])

    def test_external_asset_dependencies_is_not_checked(self) -> None:
        m1 = self._create_mock_model("m1", ["external_project/datasets/ds1"])

        try:
            check_asset_dependencies([m1])
        except Exception as e:
            pytest.fail(f"External asset dependency raised an exception: {e}")

    def test_build_graph_fails_if_project_contains_cycle(self) -> None:
        m1 = self._create_mock_model("m1", ["datasets/ds1"])
        ds1 = self._create_mock_dataset("ds1", ["models/m1"])

        with pytest.raises(
            ProjectCircularDependenciesException,
        ):
            check_asset_dependencies([ds1, m1])

    def test_project_find_cycles_returns_correct_cycles(self) -> None:
        m = self._create_mock_model("m", ["datasets/a", "datasets/e"])
        a = self._create_mock_dataset("a", ["models/m", "datasets/e"])
        e = self._create_mock_dataset("e", ["datasets/s"])
        s = self._create_mock_dataset("s", ["datasets/a", "models/m"])

        graph = _build_graph([m, a, e, s])
        cycle_paths = _find_cycles(graph)
        assert len(cycle_paths) == 5

    def test_drop_independent_entities(self) -> None:
        ds2 = self._create_mock_dataset("ds2")
        ds1 = self._create_mock_dataset("ds1")
        ds3 = self._create_mock_dataset("ds3", ["datasets/ds2"])

        m3 = self._create_mock_model("m3")
        m2 = self._create_mock_model("m2", ["datasets/ds1", "models/m3"])
        m1 = self._create_mock_model("m1", ["datasets/ds2", "models/m2"])
        m4 = self._create_mock_model("m4", ["models/m3"])

        funcs = [ds2, ds3, ds1, m2, m3, m1, m4]

        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds1"),
        ) == [ds1]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds3"),
        ) == [ds2, ds3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/models/m3"),
        ) == [m3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/models/m2"),
        ) == [ds1, m2, m3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/models/m1"),
        ) == [ds2, ds1, m2, m3, m1]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/models/m4"),
        ) == [m3, m4]

    def test_drop_independent_entities_no_dependencies(self) -> None:
        ds1 = self._create_mock_dataset("ds1")
        m1 = self._create_mock_model("m1", ["datasets/ds1"])

        funcs = [ds1, m1]

        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds1"),
            keep_dependencies=False,
        ) == [ds1.drop_dependencies()]

        assert drop_independent_entities(
            funcs,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/models/m1"),
            keep_dependencies=False,
        ) == [m1.drop_dependencies()]

    def test_build_execution_plan_linear(self) -> None:
        definitions = self._create_mock_run_linear()

        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 5
        assert ops[0].sequential.dataset_build.dataset_name == "ds1"
        assert ops[1].sequential.dataset_build.dataset_name == "ds2"
        assert ops[2].sequential.dataset_build.dataset_name == "ds3"
        assert ops[3].sequential.model_train.model_version_id.value == str(
            definitions[3].version_id
        )
        assert ops[4].sequential.model_train.model_version_id.value == str(
            definitions[4].version_id
        )
        assert ops[1].sequential.dataset_build.dependency == [
            f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds1"
        ]
        assert ops[3].sequential.model_train.dependency == [
            f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds3"
        ]
        assert ops[4].sequential.model_train.dependency == [
            f"{TEST_PROJECT_FULL_NAME.path}/models/m1"
        ]

    def test_build_execution_plan_parallel(self) -> None:
        definitions = self._create_mock_run_parallel()

        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 1
        assert len(ops[0].parallel.dataset_build) == 3
        assert len(ops[0].parallel.model_train) == 2

    def test_build_execution_plan_mixed(self) -> None:
        definitions = self._create_mock_run_mixed()
        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 3
        assert ops

    def test_drop_independent_entities_execution_plan(self) -> None:
        definitions1 = self._create_mock_run_mixed()
        definitions2 = drop_independent_entities(
            definitions1,
            AssetPath.parse(f"{TEST_PROJECT_FULL_NAME.path}/datasets/ds2"),
            keep_dependencies=False,
        )
        execution_plan = build_execution_plan(definitions2)

        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 1
        assert ops[0].sequential.dataset_build.dataset_name == "ds2"

    def _create_mock_run_linear(self) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1")
        m1 = self._create_mock_model("m1", ["datasets/ds1"])

        ds1 = self._create_mock_dataset("ds1")
        ds2 = self._create_mock_dataset("ds2", ["datasets/ds1"])
        ds3 = self._create_mock_dataset("ds3", ["datasets/ds2"])
        m1 = self._create_mock_model("m1", ["datasets/ds3"])
        m2 = self._create_mock_model("m2", ["models/m1"])

        return [ds1, ds2, ds3, m1, m2]

    def _create_mock_run_parallel(self) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1")
        ds2 = self._create_mock_dataset("ds2")
        ds3 = self._create_mock_dataset("ds3")
        m1 = self._create_mock_model("m1")
        m2 = self._create_mock_model("m2")

        return [ds1, ds2, ds3, m1, m2]

    def _create_mock_run_mixed(self) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1", project_name="other-project")
        ds2 = self._create_mock_dataset("ds2")
        ds3 = self._create_mock_dataset(
            "ds3", ["other-project/datasets/ds1"], project_name="other-project"
        )
        ds4 = self._create_mock_dataset("ds4", ["datasets/ds2"])
        m1 = self._create_mock_model("m1", ["datasets/ds3", "datasets/ds4"])

        return [ds1, ds2, ds3, ds4, m1]

    @staticmethod
    def _create_mock_dataset(
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = TEST_PROJECT_FULL_NAME.project_name,
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_asset(
            AssetType.DATASET, name, dependencies, project_name
        )

    @staticmethod
    def _create_mock_model(
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = TEST_PROJECT_FULL_NAME.project_name,
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_asset(
            AssetType.MODEL, name, dependencies, project_name
        )

    @staticmethod
    def _create_mock_asset(
        asset_type: AssetType,
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = TEST_PROJECT_FULL_NAME.project_name,
        account_name: str = TEST_PROJECT_FULL_NAME.account_name,
    ) -> FunctionDefinition:
        if dependencies is None:
            dependencies = []

        dependency_paths = [
            AssetPath.parse(d).with_project_full_name(
                ProjectFullName(
                    account_name=account_name,
                    project_name=project_name,
                )
            )
            for d in dependencies
        ]

        def func() -> None:
            pass

        return FunctionDefinition(
            func=func,
            project_name=project_name,
            account_name=account_name,
            asset_type=asset_type,
            asset_name=name,
            fabric=Fabric.F_LOCAL,
            asset_dependencies=dependency_paths,
            pip_dependencies=[],
            resource_paths=[],
            assertions=[],
            version_id=uuid.uuid4(),
        )
