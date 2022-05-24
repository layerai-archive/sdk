import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, DefaultDict, List, Optional, Sequence, Set

from layerapi.api.entity.operations_pb2 import (
    DatasetBuildOperation,
    ExecutionPlan,
    ModelTrainOperation,
    Operation,
    ParallelOperation,
    SequentialOperation,
)
from layerapi.api.ids_pb2 import ModelVersionId

from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.runs import FunctionDefinition, Run
from layer.exceptions.exceptions import (
    LayerClientException,
    ProjectCircularDependenciesException,
    ProjectDependencyNotFoundException,
)


if TYPE_CHECKING:
    from networkx import DiGraph  # type: ignore


@dataclass(frozen=True)
class PlanNode:
    path: AssetPath
    dependencies: List[AssetPath]
    name: str
    id: Optional[uuid.UUID]


def build_execution_plan(run: Run) -> ExecutionPlan:
    graph = _build_directed_acyclic_graph(run.definitions)
    plan = _topological_sort_grouping(graph)
    operations = []
    for _level, ops in plan.items():
        if len(ops) > 1:
            dataset_build_operations = []
            model_train_operations = []
            for operation in ops:
                dependencies = [d.path() for d in operation.dependencies]
                if operation.path.asset_type == AssetType.MODEL:
                    model_train_operation = ModelTrainOperation(
                        model_version_id=ModelVersionId(value=str(operation.id)),
                        dependency=dependencies,
                    )
                    model_train_operations.append(model_train_operation)
                elif operation.path.asset_type == AssetType.DATASET:
                    dataset_build_operation = DatasetBuildOperation(
                        dataset_name=operation.name,
                        dependency=dependencies,
                    )
                    dataset_build_operations.append(dataset_build_operation)
                else:
                    raise LayerClientException(f"Unknown operation type. {operation}")
            operations.append(
                Operation(
                    parallel=ParallelOperation(
                        dataset_build=dataset_build_operations,
                        model_train=model_train_operations,
                    )
                )
            )
        else:
            (operation,) = ops
            dependencies = [d.path() for d in operation.dependencies]
            if operation.path.asset_type == AssetType.MODEL:
                model_train_operation = ModelTrainOperation(
                    model_version_id=ModelVersionId(value=str(operation.id)),
                    dependency=dependencies,
                )
                operations.append(
                    Operation(
                        sequential=SequentialOperation(
                            model_train=model_train_operation
                        )
                    )
                )
            elif operation.path.asset_type == AssetType.DATASET:
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


def check_entity_dependencies(definitions: Sequence[FunctionDefinition]) -> None:
    _build_directed_acyclic_graph(definitions)


def drop_independent_entities(
    definitions: Sequence[FunctionDefinition],
    target_path: AssetPath,
    *,
    keep_dependencies: bool = True,
) -> Sequence[FunctionDefinition]:
    from networkx import NodeNotFound, shortest_path

    target_entity_id = _get_entity_id(target_path)
    if keep_dependencies:
        graph = _build_directed_acyclic_graph(definitions)
        try:
            entity_ids: Set[str] = set.union(
                set(), *shortest_path(graph, target=target_entity_id).values()
            )
        except NodeNotFound:
            raise _create_not_found_exception(target_path)
    else:
        entity_ids = {target_entity_id}
    definitions = [
        f if keep_dependencies else f.drop_dependencies()
        for f in definitions
        if _get_entity_id(f.asset_path) in entity_ids
    ]
    return definitions


def _build_graph(definitions: Sequence[FunctionDefinition]) -> "DiGraph":
    from networkx import DiGraph

    graph = DiGraph()

    for func in definitions:
        _add_function_to_graph(graph, func)

    for func in definitions:
        entity_id = _get_entity_id(func.asset_path)
        for dependency_path in func.dependencies:
            dependency_entity_id = _get_entity_id(dependency_path)
            # we add connections only to other entities to build
            if dependency_entity_id in graph.nodes:
                graph.add_edge(dependency_entity_id, entity_id)

    return graph


def _build_directed_acyclic_graph(
    definitions: Sequence[FunctionDefinition],
) -> "DiGraph":
    from networkx import is_directed_acyclic_graph

    graph = _build_graph(definitions)

    if not is_directed_acyclic_graph(graph):
        cycles: List[List[AssetPath]] = _find_cycles(graph.reverse())
        stringified_paths = [_stringify_entity_cycle(cycle) for cycle in cycles]
        stringified_paths.sort()  # Ensure stability across different runs
        raise ProjectCircularDependenciesException(stringified_paths)
    return graph


def _add_function_to_graph(graph: "DiGraph", func: FunctionDefinition) -> None:
    graph.add_node(
        _get_entity_id(func.asset_path),
        node=PlanNode(
            path=func.asset_path,
            name=func.name,
            id=func.version_id,
            dependencies=func.dependencies,
        ),
    )


def _find_cycles(graph: "DiGraph") -> List[List[AssetPath]]:
    from networkx import get_node_attributes, simple_cycles

    cycle_paths: List[List[AssetPath]] = []
    entities_map = get_node_attributes(graph, "node")
    for cycle in simple_cycles(graph):
        cycle_path: List[AssetPath] = [entities_map[node] for node in cycle]
        cycle_paths.append(cycle_path)
    return cycle_paths


def _topological_sort_grouping(
    graph: "DiGraph",
) -> DefaultDict[int, List[PlanNode]]:
    _graph = graph.copy()
    res = defaultdict(list)
    level = 0
    while _graph:
        zero_in_degree = [
            vertex for vertex, degree in _graph.in_degree() if degree == 0
        ]
        res[level] = [_graph.nodes[vertex]["node"] for vertex in zero_in_degree]
        _graph.remove_nodes_from(zero_in_degree)
        level = level + 1
    return res


def _stringify_entity_cycle(entity_cycle_path: List[AssetPath]) -> str:
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


def _get_entity_id(path: AssetPath) -> str:
    if path.project_name is None:
        raise LayerClientException("Cannot plan execution with missing project name")
    return path.path()


def _create_not_found_exception(path: AssetPath) -> Exception:
    return ProjectDependencyNotFoundException(
        f"{path.path()} not found. Declare the dependency in your project",
    )
