import logging
import os
import uuid
import warnings
from collections import defaultdict, deque, namedtuple
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from layer.api.entity.operations_pb2 import (
    DatasetBuildOperation,
    ExecutionPlan,
    HyperparameterTuningOperation,
    ModelTrainOperation,
    Operation,
    ParallelOperation,
    SequentialOperation,
)
from layer.api.ids_pb2 import HyperparameterTuningId, ModelVersionId
from layer.client import DerivedDataset, LayerClientException, Model, RawDataset
from layer.common import LayerClient
from layer.exceptions.exceptions import (
    ProjectCircularDependenciesException,
    ProjectDependencyNotFoundException,
    ProjectException,
)
from layer.projects.asset import AssetType, BaseAsset
from layer.projects.entity import EntityType
from layer.user.account import Account


if TYPE_CHECKING:
    from networkx import DiGraph  # type: ignore

logger = logging.getLogger()
LeveledNode = namedtuple("LeveledNode", ["node", "level"])


@dataclass(frozen=True)
class PlanNode:
    type: Type[BaseAsset]
    name: str
    entity: BaseAsset


@dataclass(frozen=True)
class Asset:
    type: AssetType
    name: str


@dataclass(frozen=True)
class ResourcePath:
    # Local file system path of the resource (file or dir), relative to the project dir.
    # Examples: data/test.csv, users.parquet
    path: str

    def local_relative_paths(self) -> Iterator[str]:
        """
        Map path to the absolute file system paths, checking if paths exist.
        Includes single files and files in each resource directory.

        :return: iterator of absolute file paths.
        """
        file_path = os.path.relpath(self.path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"resource file or directory: {self.path} in {os.getcwd()}"
            )
        if os.path.isfile(file_path):
            yield file_path
        if os.path.isdir(file_path):
            for root, _, files in os.walk(file_path):
                for f in files:
                    dir_file_path = os.path.join(root, f)
                    yield os.path.relpath(dir_file_path)


@dataclass(frozen=True)
class Function:
    name: str
    asset: Asset
    resource_paths: Set[ResourcePath] = field(default_factory=set)


@dataclass(frozen=True)
class Project:
    """
    Provides access to projects stored in Layer. Projects are containers to organize your machine learning project assets.

    You can retrieve an instance of this object with :code:`layer.init()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Initializes a project with name "my-project"
        layer.init("my-project")

    """

    name: str
    raw_datasets: Sequence[RawDataset] = field(default_factory=list)
    derived_datasets: Sequence[DerivedDataset] = field(default_factory=list)
    models: Sequence[Model] = field(default_factory=list)
    path: Path = field(compare=False, default=Path())
    project_files_hash: str = ""
    readme: str = ""
    account: Optional[Account] = None
    _id: Optional[uuid.UUID] = None
    functions: Sequence[Function] = field(default_factory=list)

    @property
    def id(self) -> uuid.UUID:
        if self._id is None:
            raise ProjectException("project has no id defined")
        return self._id

    def with_name(self, name: str) -> "Project":
        """
        :return: A new object that has a new name but all other fields are the same.
        """
        return replace(self, name=name)

    def with_id(self, project_id: uuid.UUID) -> "Project":
        """
        :return: A new object that has a new id but all other fields are the same.
        """
        return replace(self, _id=project_id)

    def with_account(self, account: Account) -> "Project":
        return replace(self, account=account)

    def with_raw_datasets(self, raw_datasets: Iterable[RawDataset]) -> "Project":
        return replace(self, raw_datasets=list(raw_datasets))

    def with_derived_datasets(
        self, derived_datasets: Iterable[DerivedDataset]
    ) -> "Project":
        return replace(self, derived_datasets=list(derived_datasets))

    def with_models(self, models: Iterable[Model]) -> "Project":
        return replace(self, models=list(models))

    def with_path(self, path: Path) -> "Project":
        return replace(self, path=path)

    def with_files_hash(self, new_hash: str) -> "Project":
        return replace(self, project_files_hash=new_hash)

    def with_readme(self, readme: Optional[str]) -> "Project":
        return replace(self, readme=readme)

    def with_functions(self, functions: Sequence[Function]) -> "Project":
        return replace(self, functions=functions)

    def build_execution_plan(
        self,
        models_metadata: Dict[str, ModelVersionId],
        hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId],
    ) -> ExecutionPlan:
        graph = self._build_dependency_graph()
        plan = self.topological_sort_grouping(graph)
        operations = []
        for _level, ops in plan.items():
            if len(ops) > 1:
                derived_dataset_build_operations = []
                model_train_operations = []
                hyperparameter_tuning_operations = []
                for operation in ops:
                    dependencies = [d.path for d in operation.entity.dependencies]
                    if operation.type == Model:
                        if operation.name in hyperparameter_tuning_metadata:
                            hpt_operation = HyperparameterTuningOperation(
                                hyperparameter_tuning_id=hyperparameter_tuning_metadata[
                                    operation.name
                                ],
                                dependency=dependencies,
                            )
                            hyperparameter_tuning_operations.append(hpt_operation)
                        else:
                            model_train_operation = ModelTrainOperation(
                                model_version_id=models_metadata[operation.name],
                                dependency=dependencies,
                            )
                            model_train_operations.append(model_train_operation)
                    elif issubclass(operation.type, DerivedDataset):
                        dataset_build_operation = DatasetBuildOperation(
                            dataset_name=operation.name,
                            dependency=dependencies,
                        )
                        derived_dataset_build_operations.append(dataset_build_operation)
                    else:
                        raise LayerClientException(
                            f"Unknown operation type. {operation}"
                        )
                operations.append(
                    Operation(
                        parallel=ParallelOperation(
                            dataset_build=derived_dataset_build_operations,
                            model_train=model_train_operations,
                            hyperparameter_tuning=hyperparameter_tuning_operations,
                        )
                    )
                )
            else:
                (operation,) = ops
                dependencies = [d.path for d in operation.entity.dependencies]
                if operation.type == Model:
                    if operation.name in hyperparameter_tuning_metadata:
                        hpt_operation = HyperparameterTuningOperation(
                            hyperparameter_tuning_id=hyperparameter_tuning_metadata[
                                operation.name
                            ],
                            dependency=dependencies,
                        )
                        operations.append(
                            Operation(
                                sequential=SequentialOperation(
                                    hyperparameter_tuning=hpt_operation
                                )
                            )
                        )
                    else:
                        model_train_operation = ModelTrainOperation(
                            model_version_id=models_metadata[operation.name],
                            dependency=dependencies,
                        )
                        operations.append(
                            Operation(
                                sequential=SequentialOperation(
                                    model_train=model_train_operation
                                )
                            )
                        )
                elif issubclass(operation.type, DerivedDataset):
                    dataset_build_operation = DatasetBuildOperation(
                        dataset_name=operation.name,
                        dependency=dependencies,
                    )
                    operations.append(
                        Operation(
                            sequential=SequentialOperation(
                                dataset_build=dataset_build_operation
                            )
                        )
                    )
                else:
                    raise LayerClientException(f"Unknown operation type. {operation}")
        execution_plan = ExecutionPlan(operations=operations)
        return execution_plan

    @staticmethod
    def topological_sort_grouping(
        graph: "DiGraph",
    ) -> DefaultDict[int, List[PlanNode]]:
        _graph = graph.copy()
        res = defaultdict(list)
        level = 0
        while _graph:
            zero_in_degree = [
                vertex for vertex, degree in _graph.in_degree() if degree == 0
            ]
            res[level] = [
                PlanNode(vertex[0], vertex[1], _graph.nodes[vertex]["entity"])
                for vertex in zero_in_degree
            ]
            _graph.remove_nodes_from(zero_in_degree)
            level = level + 1
        return res

    def _build_dependency_graph(self) -> "DiGraph":
        from networkx import DiGraph, is_directed_acyclic_graph

        graph = DiGraph()
        entities: Sequence[BaseAsset] = [
            *self.derived_datasets,
            *self.models,
        ]

        for entity in entities:
            name = self._get_entity_id(entity)
            graph.add_node(name, entity=entity)
        for entity in entities:
            entity_id = self._get_entity_id(entity)
            for dependency in entity.dependencies:
                dependency_entity_id = self._get_entity_id(dependency)
                # we add connections only to other entities to build
                for node_entity_id in graph.nodes:
                    if self._is_same_entity(dependency_entity_id, node_entity_id):
                        graph.add_edge(node_entity_id, entity_id)

        if not is_directed_acyclic_graph(graph):
            cycles: List[List[BaseAsset]] = self.find_cycles(graph.reverse())
            stringified_paths = [
                self._stringify_entity_cycle(cycle) for cycle in cycles
            ]
            stringified_paths.sort()  # Ensure stability across different runs
            raise ProjectCircularDependenciesException(stringified_paths)
        return graph

    @staticmethod
    def _is_same_entity(
        dependency_entity_id: Tuple[Type[BaseAsset], str],
        node_entity_id: Tuple[Type[BaseAsset], str],
    ) -> bool:
        (dependency_type, dependency_name) = dependency_entity_id
        (node_type, node_name) = node_entity_id
        # graph is build from concrete classes, while dependencies are specified by users using abstract
        # classes. we need to account for that
        return issubclass(node_type, dependency_type) and node_name == dependency_name

    @staticmethod
    def find_cycles(graph: "DiGraph") -> List[List[BaseAsset]]:
        from networkx import get_node_attributes, simple_cycles

        cycle_paths: List[List[BaseAsset]] = []
        entities_map = get_node_attributes(graph, "entity")
        for cycle in simple_cycles(graph):
            cycle_path: List[BaseAsset] = [entities_map[node] for node in cycle]
            cycle_paths.append(cycle_path)
        return cycle_paths

    @staticmethod
    def _stringify_entity_cycle(entity_cycle_path: List[BaseAsset]) -> str:
        def rotate_list(entity_cycle_path: List[str]) -> List[str]:
            smallest_idx = 0
            for i in range(1, len(entity_cycle_path)):
                if entity_cycle_path[i] < entity_cycle_path[smallest_idx]:
                    smallest_idx = i
            rotated = deque(entity_cycle_path)
            rotated.rotate(-smallest_idx)
            return list(rotated)

        stringified = rotate_list(
            [str(entity) for entity in entity_cycle_path]
        )  # Ensure stability within a cycle path across runs
        stringified.append(stringified[0])  # Add dependency to first node in cycle
        return " -> ".join(stringified)

    @staticmethod
    def _get_entity_id(entity: BaseAsset) -> Tuple[Type[BaseAsset], str]:
        return type(entity), entity.name

    @staticmethod
    def _create_not_found_exception(
        entity_id: Tuple[Type[BaseAsset], str]
    ) -> Exception:
        type_, name = entity_id
        return ProjectDependencyNotFoundException(
            f"{type_.__name__.lower().capitalize()} {name!r} not found",
            "Declare the dependency in your project",
        )

    def check_entity_dependencies(self) -> None:
        self._build_dependency_graph()

    def drop_independent_entities(
        self, type_: "EntityType", name: "str", *, keep_dependencies: bool = True
    ) -> "Project":
        from networkx import NodeNotFound, shortest_path

        target_entity_id = type_.get_factory(), name
        graph = self._build_dependency_graph()
        try:
            # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            entity_ids: Set[Tuple[Type[BaseAsset], str]] = set.union(
                set(), *shortest_path(graph, target=target_entity_id).values()
            )
        except NodeNotFound:
            raise self._create_not_found_exception(target_entity_id)
        if not keep_dependencies:
            entity_ids = {target_entity_id}
        raw_datasets = [e for e in self.raw_datasets if keep_dependencies]
        derived_datasets = [
            e if keep_dependencies else e.drop_dependencies()
            for e in self.derived_datasets
            if self._get_entity_id(e) in entity_ids
        ]
        models = [
            e if keep_dependencies else e.drop_dependencies()
            for e in self.models
            if self._get_entity_id(e) in entity_ids
        ]
        return (
            self.with_raw_datasets(raw_datasets)
            .with_derived_datasets(derived_datasets)
            .with_models(models)
        )


@dataclass(frozen=True)
class ApplyResult:
    execution_plan: ExecutionPlan
    models_metadata: Dict[str, ModelVersionId] = field(default_factory=dict)
    hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId] = field(
        default_factory=dict
    )

    def with_models_metadata(
        self,
        models_metadata: Dict[str, ModelVersionId],
        hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId],
        plan: ExecutionPlan,
    ) -> "ApplyResult":
        return replace(
            self,
            models_metadata=dict(models_metadata),
            hyperparameter_tuning_metadata=dict(hyperparameter_tuning_metadata),
            execution_plan=plan,
        )


class ProjectLoader:
    @staticmethod
    def load_project_readme(path: Path) -> Optional[str]:
        for subp in path.glob("*"):
            if subp.name.lower() == "readme.md":
                with open(subp, "r") as f:
                    readme = f.read()
                    # restrict length of text we send to backend
                    if len(readme) > 25_000:
                        warnings.warn(
                            "Your README.md will be truncated to 25000 characters",
                        )
                        readme = readme[:25_000]
                    return readme
        return None


def get_or_create_remote_project(client: LayerClient, project: Project) -> Project:
    project_id_with_org_id = client.project_service_client.get_project_id_and_org_id(
        project.name
    )
    if project_id_with_org_id.project_id is None:
        project_id_with_org_id = client.project_service_client.create_project(
            project.name
        )
    assert project_id_with_org_id.project_id is not None
    assert project_id_with_org_id.account_id is not None
    account_name = client.account.get_account_name_by_id(
        project_id_with_org_id.account_id
    )
    account = Account(id=project_id_with_org_id.account_id, name=account_name)
    return project.with_id(project_id_with_org_id.project_id).with_account(
        account=account
    )
