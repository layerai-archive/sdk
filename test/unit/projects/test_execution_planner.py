import uuid
from typing import Optional, Sequence, Type

import pytest

from layer.contracts.asset import AssetPath
from layer.contracts.fabrics import Fabric
from layer.contracts.runs import (
    DatasetFunctionDefinition,
    FunctionDefinition,
    ModelFunctionDefinition,
    Run,
)
from layer.exceptions.exceptions import ProjectCircularDependenciesException
from layer.projects.execution_planner import (
    _build_graph,
    _find_cycles,
    build_execution_plan,
    check_asset_dependencies,
    drop_independent_entities,
)
from layer.settings import LayerSettings


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
            pytest.fail(f"External entity dependency raised an exception: {e}")

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
            AssetPath.parse("test/datasets/ds1"),
        ) == [ds1]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse("test/datasets/ds3"),
        ) == [ds2, ds3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse("test/models/m3"),
        ) == [m3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse("test/models/m2"),
        ) == [ds1, m2, m3]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse("test/models/m1"),
        ) == [ds2, ds1, m2, m3, m1]
        assert drop_independent_entities(
            funcs,
            AssetPath.parse("test/models/m4"),
        ) == [m3, m4]

    def test_drop_independent_entities_no_dependencies(self) -> None:
        ds1 = self._create_mock_dataset("ds1")
        m1 = self._create_mock_model("m1", ["datasets/ds1"])

        funcs = [ds1, m1]

        assert drop_independent_entities(
            funcs, AssetPath.parse("test/datasets/ds1"), keep_dependencies=False
        ) == [ds1.drop_dependencies()]

        assert drop_independent_entities(
            funcs, AssetPath.parse("test/models/m1"), keep_dependencies=False
        ) == [m1.drop_dependencies()]

    def test_build_execution_plan_linear(self) -> None:
        r1 = self._create_mock_run_linear()

        execution_plan = build_execution_plan(r1)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 5
        assert ops[0].sequential.dataset_build.dataset_name == "ds1"
        assert ops[1].sequential.dataset_build.dataset_name == "ds2"
        assert ops[2].sequential.dataset_build.dataset_name == "ds3"
        assert ops[3].sequential.model_train.model_version_id.value == str(
            r1.definitions[3].version_id
        )
        assert ops[4].sequential.model_train.model_version_id.value == str(
            r1.definitions[4].version_id
        )
        assert ops[1].sequential.dataset_build.dependency == ["test/datasets/ds1"]
        assert ops[3].sequential.model_train.dependency == ["test/datasets/ds3"]
        assert ops[4].sequential.model_train.dependency == ["test/models/m1"]

    def test_build_execution_plan_parallel(self) -> None:
        r1 = self._create_mock_run_parallel()

        execution_plan = build_execution_plan(r1)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 1
        assert len(ops[0].parallel.dataset_build) == 3
        assert len(ops[0].parallel.model_train) == 2

    def test_build_execution_plan_mixed(self) -> None:
        r1 = self._create_mock_run_mixed()
        execution_plan = build_execution_plan(r1)
        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 3
        assert ops

    def test_drop_independent_entities_execution_plan(self) -> None:
        r1 = self._create_mock_run_mixed()
        r2 = r1.with_definitions(
            drop_independent_entities(
                r1.definitions,
                AssetPath.parse("test/datasets/ds2"),
                keep_dependencies=False,
            )
        )
        execution_plan = build_execution_plan(r2)

        assert execution_plan is not None
        ops = execution_plan.operations

        assert len(ops) == 1
        assert ops[0].sequential.dataset_build.dataset_name == "ds2"

    def _create_mock_run_linear(self) -> Run:
        ds1 = self._create_mock_dataset("ds1")
        m1 = self._create_mock_model("m1", ["datasets/ds1"])

        ds1 = self._create_mock_dataset("ds1")
        ds2 = self._create_mock_dataset("ds2", ["datasets/ds1"])
        ds3 = self._create_mock_dataset("ds3", ["datasets/ds2"])
        m1 = self._create_mock_model("m1", ["datasets/ds3"])
        m2 = self._create_mock_model("m2", ["models/m1"])

        return Run(uuid.uuid4(), "test", definitions=[ds1, ds2, ds3, m1, m2])

    def _create_mock_run_parallel(self) -> Run:
        ds1 = self._create_mock_dataset("ds1")
        ds2 = self._create_mock_dataset("ds2")
        ds3 = self._create_mock_dataset("ds3")
        m1 = self._create_mock_model("m1")
        m2 = self._create_mock_model("m2")

        return Run(uuid.uuid4(), "test", definitions=[ds1, ds2, ds3, m1, m2])

    def _create_mock_run_mixed(self) -> Run:
        ds1 = self._create_mock_dataset("ds1", project_name="other-project")
        ds2 = self._create_mock_dataset("ds2")
        ds3 = self._create_mock_dataset(
            "ds3", ["other-project/datasets/ds1"], project_name="other-project"
        )
        ds4 = self._create_mock_dataset("ds4", ["datasets/ds2"])
        m1 = self._create_mock_model("m1", ["datasets/ds3", "datasets/ds4"])

        return Run(uuid.uuid4(), "test", definitions=[ds1, ds2, ds3, ds4, m1])

    @staticmethod
    def _create_mock_dataset(
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = "test",
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_entity(
            DatasetFunctionDefinition, name, dependencies, project_name
        )

    @staticmethod
    def _create_mock_model(
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = "test",
    ) -> FunctionDefinition:
        return TestProjectExecutionPlanner._create_mock_entity(
            ModelFunctionDefinition, name, dependencies, project_name
        )

    @staticmethod
    def _create_mock_entity(
        asset_type: Type[FunctionDefinition],
        name: str,
        dependencies: Optional[Sequence[str]] = None,
        project_name: str = "test",
    ) -> FunctionDefinition:
        if dependencies is None:
            dependencies = []

        dependency_paths = []
        for dependency in dependencies:
            path = AssetPath.parse(dependency)
            if path.project_name is None:
                path = path.with_project_name(project_name)
            dependency_paths.append(path)

        def func() -> None:
            pass

        settings = LayerSettings()
        settings.set_asset_name(name)
        settings.set_dependencies(dependency_paths)
        settings.set_fabric(Fabric.F_LOCAL.value)

        func.layer = settings  # type: ignore

        return asset_type(func, project_name, version_id=uuid.uuid4())
