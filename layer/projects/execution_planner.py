from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Sequence, Set, Tuple, Type

from layerapi.api.entity.operations_pb2 import (
    DatasetBuildOperation,
    ExecutionPlan,
    HyperparameterTuningOperation,
    ModelTrainOperation,
    Operation,
    ParallelOperation,
    SequentialOperation,
)
from layerapi.api.ids_pb2 import HyperparameterTuningId, ModelVersionId

from layer.contracts.asset import BaseAsset
from layer.contracts.datasets import DerivedDataset
from layer.contracts.entities import EntityType
from layer.contracts.models import Model
from layer.contracts.projects import Project
from layer.exceptions.exceptions import (
    LayerClientException,
    ProjectCircularDependenciesException,
    ProjectDependencyNotFoundException,
)


if TYPE_CHECKING:
    from networkx import DiGraph  # type: ignore


@dataclass(frozen=True)
class PlanNode:
    type: Type[BaseAsset]
    name: str
    entity: BaseAsset


def build_execution_plan(
    project: Project,
    models_metadata: Dict[str, ModelVersionId],
    hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId],
) -> ExecutionPlan:
    graph = _build_dependency_graph(project)
    plan = topological_sort_grouping(graph)
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
                    raise LayerClientException(f"Unknown operation type. {operation}")
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


def _build_dependency_graph(project: Project) -> "DiGraph":
    from networkx import DiGraph, is_directed_acyclic_graph

    graph = DiGraph()
    entities: Sequence[BaseAsset] = [
        *project.derived_datasets,
        *project.models,
    ]

    for entity in entities:
        name = _get_entity_id(entity)
        graph.add_node(name, entity=entity)
    for entity in entities:
        entity_id = _get_entity_id(entity)
        for dependency in entity.dependencies:
            dependency_entity_id = _get_entity_id(dependency)
            # we add connections only to other entities to build
            for node_entity_id in graph.nodes:
                if _is_same_entity(dependency_entity_id, node_entity_id):
                    graph.add_edge(node_entity_id, entity_id)

    if not is_directed_acyclic_graph(graph):
        cycles: List[List[BaseAsset]] = find_cycles(graph.reverse())
        stringified_paths = [_stringify_entity_cycle(cycle) for cycle in cycles]
        stringified_paths.sort()  # Ensure stability across different runs
        raise ProjectCircularDependenciesException(stringified_paths)
    return graph


def _is_same_entity(
    dependency_entity_id: Tuple[Type[BaseAsset], str],
    node_entity_id: Tuple[Type[BaseAsset], str],
) -> bool:
    (dependency_type, dependency_name) = dependency_entity_id
    (node_type, node_name) = node_entity_id
    # graph is build from concrete classes, while dependencies are specified by users using abstract
    # classes. we need to account for that
    return issubclass(node_type, dependency_type) and node_name == dependency_name


def find_cycles(graph: "DiGraph") -> List[List[BaseAsset]]:
    from networkx import get_node_attributes, simple_cycles

    cycle_paths: List[List[BaseAsset]] = []
    entities_map = get_node_attributes(graph, "entity")
    for cycle in simple_cycles(graph):
        cycle_path: List[BaseAsset] = [entities_map[node] for node in cycle]
        cycle_paths.append(cycle_path)
    return cycle_paths


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


def _get_entity_id(entity: BaseAsset) -> Tuple[Type[BaseAsset], str]:
    return type(entity), entity.name


def check_entity_dependencies(project: Project) -> None:
    _build_dependency_graph(project)


def drop_independent_entities(
    project: Project,
    type_: EntityType,
    name: str,
    *,
    keep_dependencies: bool = True,
) -> "Project":
    from networkx import NodeNotFound, shortest_path

    target_entity_id = type_.get_factory(), name
    graph = _build_dependency_graph(project)
    try:
        entity_ids: Set[Tuple[Type[BaseAsset], str]] = set.union(
            set(), *shortest_path(graph, target=target_entity_id).values()
        )
    except NodeNotFound:
        raise _create_not_found_exception(target_entity_id)
    if not keep_dependencies:
        entity_ids = {target_entity_id}
    raw_datasets = [e for e in project.raw_datasets if keep_dependencies]
    derived_datasets = [
        e if keep_dependencies else e.drop_dependencies()
        for e in project.derived_datasets
        if _get_entity_id(e) in entity_ids
    ]
    models = [
        e if keep_dependencies else e.drop_dependencies()
        for e in project.models
        if _get_entity_id(e) in entity_ids
    ]
    return (
        project.with_raw_datasets(raw_datasets)
        .with_derived_datasets(derived_datasets)
        .with_models(models)
    )


def _create_not_found_exception(entity_id: Tuple[Type[BaseAsset], str]) -> Exception:
    type_, name = entity_id
    return ProjectDependencyNotFoundException(
        f"{type_.__name__.lower().capitalize()} {name!r} not found",
        "Declare the dependency in your project",
    )
