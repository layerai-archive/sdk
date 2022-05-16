import uuid
from pathlib import Path
from typing import Dict, List

import pytest

from layer.api.ids_pb2 import HyperparameterTuningId, ModelVersionId
from layer.data_classes import (
    Dataset,
    DerivedDataset,
    Model,
    Parameter,
    PythonDataset,
    RawDataset,
    Train,
)
from layer.exceptions.exceptions import (
    ProjectCircularDependenciesException,
    ProjectException,
)
from layer.projects.asset import AssetPath, AssetType
from layer.projects.entity import EntityType
from layer.projects.project import Project


class TestProject:
    def test_project_id_property_throws_when_absent(self) -> None:
        with pytest.raises(ProjectException, match="project has no id defined"):
            project = Project(name="test")

            _ = project.id

    def test_project_id_property_returns_when_present(self) -> None:
        project = Project(name="test")

        new_id = uuid.uuid4()
        project = project.with_id(new_id)

        assert new_id == project.id

    def test_check_entity_external_dependency(self) -> None:
        fs1 = self._create_mock_derived_dataset("fs1", project_name="another-project-x")

        m1 = self._create_mock_model("m1", [fs1])

        project = Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[],
            models=[m1],
        )

        project.check_entity_dependencies()

    def test_external_entity_dependencies_is_not_checked(self) -> None:
        fs1 = self._create_mock_derived_dataset("external_project/datasets/fs1")

        m1 = self._create_mock_model("m1", [fs1])

        project = Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[],
            models=[m1],
        )

        try:
            project.check_entity_dependencies()
        except Exception as e:
            pytest.fail(f"External entity dependency raised an exception: {e}")

    def test_build_graph_fails_if_project_contains_cycle(self):
        fs1 = self._create_mock_derived_dataset("fs1")
        m1 = self._create_mock_model("m1", [fs1])
        fs1 = self._create_mock_derived_dataset("fs1", [m1])

        project = Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[fs1],
            models=[m1],
        )

        with pytest.raises(
            ProjectCircularDependenciesException,
        ):
            project.check_entity_dependencies()

    def test_project_find_cycles_returns_correct_cycles(self) -> None:
        from networkx import DiGraph

        graph = DiGraph()
        m = self._create_mock_model("m")
        a = self._create_mock_derived_dataset("a")
        e = self._create_mock_derived_dataset("e")
        s = self._create_mock_derived_dataset("s")
        m = m.with_dependencies([a, e])
        a = a.with_dependencies([m, e])
        e = e.with_dependencies([s])
        s = s.with_dependencies([a, m])
        entities = [m, a, e, s]

        for entity in entities:
            graph.add_node(str(entity), entity=entity)
        for entity in entities:
            entity_id = str(entity)
            for dependency in entity.dependencies:
                dependency_id = str(dependency)
                graph.add_edge(dependency_id, entity_id)

        cycle_paths = Project.find_cycles(graph)
        assert len(cycle_paths) == 5

    def test_drop_independent_entities(self) -> None:
        ds1 = self._create_mock_raw_dataset("ds1")

        fs2 = self._create_mock_derived_dataset("fs2")
        fs1 = self._create_mock_derived_dataset("fs1")
        fs3 = self._create_mock_derived_dataset("fs3", [fs2])

        m3 = self._create_mock_model("m3")
        m2 = self._create_mock_model("m2", [fs1, m3])
        m1 = self._create_mock_model("m1", [m2, fs2])
        m4 = self._create_mock_model("m4", [m3])

        project = Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs2, fs3, fs1],
            models=[m2, m3, m1, m4],
        )

        assert project.drop_independent_entities(
            EntityType.DERIVED_DATASET, "fs1"
        ) == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs1],
            models=[],
        )
        assert project.drop_independent_entities(
            EntityType.DERIVED_DATASET, "fs3"
        ) == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs2, fs3],
            models=[],
        )
        assert project.drop_independent_entities(EntityType.MODEL, "m3") == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[],
            models=[m3],
        )
        assert project.drop_independent_entities(EntityType.MODEL, "m2") == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs1],
            models=[m2, m3],
        )
        assert project.drop_independent_entities(EntityType.MODEL, "m1") == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs2, fs1],
            models=[m2, m3, m1],
        )
        assert project.drop_independent_entities(EntityType.MODEL, "m4") == Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[],
            models=[m3, m4],
        )

    def test_drop_independent_entities_no_dependencies(self) -> None:
        ds1 = self._create_mock_raw_dataset("ds1")

        fs1 = self._create_mock_derived_dataset("fs1")

        m1 = self._create_mock_model("m1", [fs1])

        project = Project(
            name="test",
            raw_datasets=[ds1],
            derived_datasets=[fs1],
            models=[m1],
        )

        assert project.drop_independent_entities(
            EntityType.DERIVED_DATASET, "fs1", keep_dependencies=False
        ) == Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[fs1.drop_dependencies()],
            models=[],
        )

        assert project.drop_independent_entities(
            EntityType.MODEL, "m1", keep_dependencies=False
        ) == Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[],
            models=[m1.drop_dependencies()],
        )

    def test_build_execution_plan_linear(self) -> None:
        p1 = self._create_mock_project_linear()
        execution_plan = p1.build_execution_plan(
            self._create_mock_model_metadata(["m1"]),
            self._create_mock_hpt_model_metadata(["hpt1"]),
        )
        assert execution_plan is not None
        assert len(execution_plan.operations) == 5
        assert (
            execution_plan.operations[0].sequential.dataset_build.dataset_name == "fs1"
        )
        assert (
            execution_plan.operations[1].sequential.dataset_build.dataset_name == "fs2"
        )
        assert (
            execution_plan.operations[2].sequential.dataset_build.dataset_name == "fs3"
        )
        assert (
            execution_plan.operations[3].sequential.model_train.model_version_id.value
            == "dummy-version-id"
        )
        assert (
            execution_plan.operations[
                4
            ].sequential.hyperparameter_tuning.hyperparameter_tuning_id.value
            == "dummy_hpt_id"
        )

    def test_build_execution_plan_with_deps_linear(self) -> None:
        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            entity_name="d1",
            project_name="test",
        )
        derived_ds = PythonDataset(asset_path=asset_path)
        d1 = Dataset("d1")

        m1 = self._create_mock_model("m1", [d1])
        p1 = (
            Project(name="test")
            .with_derived_datasets(derived_datasets=[derived_ds])
            .with_models(models=[m1])
            .with_files_hash("")
        )

        execution_plan = p1.build_execution_plan(
            self._create_mock_model_metadata(["m1"]), {}
        )

        assert execution_plan is not None
        assert len(execution_plan.operations) == 2
        assert (
            execution_plan.operations[0].sequential.dataset_build.dataset_name == "d1"
        )
        assert execution_plan.operations[1].sequential.model_train.dependency == [
            "datasets/d1"
        ]

    def test_build_execution_plan_parallel(self) -> None:
        p1 = self._create_mock_project_parallel()
        execution_plan = p1.build_execution_plan(
            self._create_mock_model_metadata(["m1"]),
            self._create_mock_hpt_model_metadata(["hpt1"]),
        )
        assert execution_plan is not None
        assert len(execution_plan.operations) == 1
        assert len(execution_plan.operations[0].parallel.dataset_build) == 3
        assert len(execution_plan.operations[0].parallel.model_train) == 1
        assert len(execution_plan.operations[0].parallel.hyperparameter_tuning) == 1

    def test_build_execution_plan_mixed(self) -> None:
        p1 = self._create_mock_project_mixed()
        execution_plan = p1.build_execution_plan(
            self._create_mock_model_metadata(["m1"]), {}
        )
        assert execution_plan is not None
        assert len(execution_plan.operations) == 3
        assert execution_plan.operations

    def test_drop_independent_entities_execution_plan(self) -> None:
        project = self._create_mock_project_mixed()

        p1 = project.drop_independent_entities(
            EntityType.DERIVED_DATASET, "fs1", keep_dependencies=False
        )
        execution_plan = p1.build_execution_plan(
            self._create_mock_model_metadata(["m1"]), {}
        )

        assert execution_plan is not None
        assert len(execution_plan.operations) == 1
        assert (
            execution_plan.operations[0].sequential.dataset_build.dataset_name == "fs1"
        )

    def _create_mock_project_linear(self) -> Project:
        fs1 = self._create_mock_derived_dataset("fs1")
        fs2 = self._create_mock_derived_dataset("fs2", [fs1])
        fs3 = self._create_mock_derived_dataset("fs3", [fs2])
        m1 = self._create_mock_model("m1", [fs3])
        hpt1 = self._create_mock_model("hpt1", [m1])
        return Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[fs1, fs2, fs3],
            models=[m1, hpt1],
        )

    def _create_mock_project_parallel(self) -> Project:
        fs1 = self._create_mock_derived_dataset("fs1")
        fs2 = self._create_mock_derived_dataset("fs2")
        fs3 = self._create_mock_derived_dataset("fs3")
        m1 = self._create_mock_model("m1")
        hpt1 = self._create_mock_model("hpt1")
        return Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[fs1, fs2, fs3],
            models=[m1, hpt1],
        )

    def _create_mock_project_mixed(self) -> Project:
        fs1 = self._create_mock_derived_dataset("fs1", project_name="other-project")
        fs2 = self._create_mock_derived_dataset("fs2")
        fs3 = self._create_mock_derived_dataset(
            "fs3", [fs1], project_name="other-project"
        )
        fs4 = self._create_mock_derived_dataset("fs4", [fs2])
        m1 = self._create_mock_model("m1", [fs3, fs4])
        return Project(
            name="test",
            raw_datasets=[],
            derived_datasets=[fs1, fs2, fs3, fs4],
            models=[m1],
        )

    @staticmethod
    def _create_mock_hpt_model_metadata(
        hpt_model_names: List[str],
    ) -> Dict[str, HyperparameterTuningId]:
        hpt_models_metadata = {}
        for name in hpt_model_names:
            hpt_models_metadata[name] = HyperparameterTuningId(value="dummy_hpt_id")
        return hpt_models_metadata

    @staticmethod
    def _create_mock_model_metadata(
        model_names: List[str],
    ) -> Dict[str, ModelVersionId]:
        models_metadata = {}
        for name in model_names:
            models_metadata[name] = ModelVersionId(value="dummy-version-id")
        return models_metadata

    @staticmethod
    def _create_mock_model(name: str, dependencies=None, project_name="test"):
        if dependencies is None:
            dependencies = []
        asset_path = AssetPath(
            entity_name=name,
            asset_type=AssetType.MODEL,
            project_name=project_name,
        )
        return Model(
            asset_path=asset_path,
            dependencies=dependencies,
            description="",
            local_path=Path.cwd(),
            training=Train(
                name="test",
                description="foo",
                entrypoint="car_model.py",
                environment="requirements.txt",
                parameters=[
                    Parameter(name="first", value="1"),
                ],
            ),
            training_files_digest="bar",
        )

    @staticmethod
    def _create_mock_derived_dataset(
        name: str, dependencies=None, project_name: str = "test-project"
    ) -> DerivedDataset:
        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            entity_name=name,
            project_name=project_name,
        )
        return DerivedDataset(asset_path=asset_path, dependencies=dependencies)

    @staticmethod
    def _create_mock_raw_dataset(
        name: str, project_name: str = "test-project"
    ) -> RawDataset:
        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            entity_name=name,
            project_name=project_name,
        )
        return RawDataset(asset_path=asset_path)
