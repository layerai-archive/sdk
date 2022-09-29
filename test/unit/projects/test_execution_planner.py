from typing import List, Optional, Sequence

import pytest

from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.exceptions.exceptions import ProjectCircularDependenciesException
from layer.projects.execution_planner import (
    _build_graph,
    _find_cycles,
    build_execution_plan,
    check_asset_dependencies,
)


class TestProjectExecutionPlanner:
    def test_check_asset_external_dependency(self, project_name: str) -> None:
        ds1 = self._create_mock_dataset("ds1", project_name="another-project-x")
        m1 = self._create_mock_model(
            "m1",
            project_name=project_name,
            dependencies=["another-project-x/datasets/ds1"],
        )

        check_asset_dependencies([m1, ds1])

    def test_external_asset_dependencies_is_not_checked(
        self, project_name: str
    ) -> None:
        m1 = self._create_mock_model(
            "m1",
            project_name=project_name,
            dependencies=["external_project/datasets/ds1"],
        )

        try:
            check_asset_dependencies([m1])
        except Exception as e:
            pytest.fail(f"External asset dependency raised an exception: {e}")

    def test_build_graph_fails_if_project_contains_cycle(
        self, project_name: str
    ) -> None:
        m1 = self._create_mock_model(
            "m1", project_name=project_name, dependencies=["datasets/ds1"]
        )
        ds1 = self._create_mock_dataset(
            "ds1", project_name=project_name, dependencies=["models/m1"]
        )

        with pytest.raises(
            ProjectCircularDependenciesException,
        ):
            check_asset_dependencies([ds1, m1])

    def test_project_find_cycles_returns_correct_cycles(
        self, project_name: str
    ) -> None:
        m = self._create_mock_model(
            "m",
            project_name=project_name,
            dependencies=["datasets/a", "datasets/e"],
        )
        a = self._create_mock_dataset(
            "a", project_name=project_name, dependencies=["models/m", "datasets/e"]
        )
        e = self._create_mock_dataset(
            "e", project_name=project_name, dependencies=["datasets/s"]
        )
        s = self._create_mock_dataset(
            "s", project_name=project_name, dependencies=["datasets/a", "models/m"]
        )

        graph = _build_graph([m, a, e, s])
        cycle_paths = _find_cycles(graph)
        assert len(cycle_paths) == 5

    def test_build_execution_plan_linear(self, project_name: str) -> None:
        definitions = self._create_mock_run_linear(project_name)

        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 5
        assert (
            ops[0].sequential.function_execution.asset_name
            == f"test-acc-from-conftest/{project_name}/datasets/ds1"
        )
        assert (
            ops[1].sequential.function_execution.asset_name
            == f"test-acc-from-conftest/{project_name}/datasets/ds2"
        )
        assert (
            ops[2].sequential.function_execution.asset_name
            == f"test-acc-from-conftest/{project_name}/datasets/ds3"
        )
        assert ops[1].sequential.function_execution.dependency == [
            f"test-acc-from-conftest/{project_name}/datasets/ds1"
        ]
        assert ops[3].sequential.function_execution.dependency == [
            f"test-acc-from-conftest/{project_name}/datasets/ds3"
        ]
        assert ops[4].sequential.function_execution.dependency == [
            f"test-acc-from-conftest/{project_name}/models/m1"
        ]

    def test_build_execution_plan_parallel(self, project_name: str) -> None:
        definitions = self._create_mock_run_parallel(project_name)

        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 1
        assert len(ops[0].parallel.function_execution) == 5

    def test_build_execution_plan_mixed(self, project_name: str) -> None:
        definitions = self._create_mock_run_mixed(project_name)
        execution_plan = build_execution_plan(definitions)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 3
        assert ops

    def _create_mock_run_linear(self, project_name: str) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1", project_name=project_name)
        m1 = self._create_mock_model(
            "m1", project_name=project_name, dependencies=["datasets/ds1"]
        )

        ds1 = self._create_mock_dataset("ds1", project_name=project_name)
        ds2 = self._create_mock_dataset(
            "ds2", project_name=project_name, dependencies=["datasets/ds1"]
        )
        ds3 = self._create_mock_dataset(
            "ds3", project_name=project_name, dependencies=["datasets/ds2"]
        )
        m1 = self._create_mock_model(
            "m1", project_name=project_name, dependencies=["datasets/ds3"]
        )
        m2 = self._create_mock_model(
            "m2", project_name=project_name, dependencies=["models/m1"]
        )

        return [ds1, ds2, ds3, m1, m2]

    def _create_mock_run_parallel(self, project_name: str) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1", project_name=project_name)
        ds2 = self._create_mock_dataset("ds2", project_name=project_name)
        ds3 = self._create_mock_dataset("ds3", project_name=project_name)
        m1 = self._create_mock_model("m1", project_name=project_name)
        m2 = self._create_mock_model("m2", project_name=project_name)

        return [ds1, ds2, ds3, m1, m2]

    def _create_mock_run_mixed(self, project_name: str) -> List[FunctionDefinition]:
        ds1 = self._create_mock_dataset("ds1", project_name="other-project")
        ds2 = self._create_mock_dataset("ds2", project_name=project_name)
        ds3 = self._create_mock_dataset(
            "ds3",
            project_name="other-project",
            dependencies=["other-project/datasets/ds1"],
        )
        ds4 = self._create_mock_dataset(
            "ds4", project_name=project_name, dependencies=["datasets/ds2"]
        )
        m1 = self._create_mock_model(
            "m1",
            project_name=project_name,
            dependencies=["datasets/ds3", "datasets/ds4"],
        )

        return [ds1, ds2, ds3, ds4, m1]

    @staticmethod
    def _create_mock_dataset(
        name: str,
        project_name: str,
        dependencies: Optional[Sequence[str]] = None,
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_asset(
            asset_type=AssetType.DATASET,
            name=name,
            project_name=project_name,
            dependencies=dependencies,
        )

    @staticmethod
    def _create_mock_model(
        name: str,
        project_name: str,
        dependencies: Optional[Sequence[str]] = None,
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_asset(
            asset_type=AssetType.MODEL,
            name=name,
            project_name=project_name,
            dependencies=dependencies,
        )

    @staticmethod
    def _create_mock_asset(
        asset_type: AssetType,
        name: str,
        project_name: str,
        account_name: str = "test-acc-from-conftest",
        dependencies: Optional[Sequence[str]] = None,
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
            args=tuple(),
            kwargs={},
            asset_type=asset_type,
            asset_name=name,
            fabric=Fabric.F_LOCAL,
            asset_dependencies=dependency_paths,
            pip_dependencies=[],
            conda_env=None,
            resource_paths=[],
            assertions=[],
        )
