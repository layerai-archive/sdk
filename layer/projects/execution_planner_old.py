import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, DefaultDict, List, Optional, Sequence

from layerapi.api.entity.operations_pb2 import (
    DatasetBuildOperation,
    ExecutionPlan,
    ModelTrainOperation,
    Operation,
    ParallelOperation,
    SequentialOperation,
)
from layerapi.api.ids_pb2 import ModelVersionId

from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.definitions import FunctionDefinition
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


def build_execution_plan(definitions: Sequence[FunctionDefinition]) -> ExecutionPlan:
    graph = _build_directed_acyclic_graph(definitions)
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


def check_asset_dependencies(definitions: Sequence[FunctionDefinition]) -> None:
    _build_directed_acyclic_graph(definitions)


def _build_graph(definitions: Sequence[FunctionDefinition]) -> "DiGraph":
    from networkx import DiGraph

    graph = DiGraph()

    for func in definitions:
        _add_function_to_graph(graph, func)

    for func in definitions:
        asset_id = _get_asset_id(func.asset_path)
        for dependency_path in func.asset_dependencies:
            dependency_asset_id = _get_asset_id(dependency_path)
            # we add connections only to other entities to build
            if dependency_asset_id in graph.nodes:
                graph.add_edge(dependency_asset_id, asset_id)

    return graph


def _build_directed_acyclic_graph(
    definitions: Sequence[FunctionDefinition],
) -> "DiGraph":
    from networkx import is_directed_acyclic_graph

    graph = _build_graph(definitions)

    if not is_directed_acyclic_graph(graph):
        cycles: List[List[AssetPath]] = _find_cycles(graph.reverse())
        stringified_paths = [_stringify_asset_cycle(cycle) for cycle in cycles]
        stringified_paths.sort()  # Ensure stability across different runs
        raise ProjectCircularDependenciesException(stringified_paths)
    return graph


def _add_function_to_graph(graph: "DiGraph", func: FunctionDefinition) -> None:
    graph.add_node(
        _get_asset_id(func.asset_path),
        node=PlanNode(
            path=func.asset_path,
            name=func.asset_name,
            id=func.version_id,
            dependencies=func.asset_dependencies,
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


def _stringify_asset_cycle(asset_cycle_path: List[AssetPath]) -> str:
    def rotate_list(asset_cycle_path: List[str]) -> List[str]:
        smallest_idx = 0
        for i in range(1, len(asset_cycle_path)):
            if asset_cycle_path[i] < asset_cycle_path[smallest_idx]:
                smallest_idx = i
        rotated = deque(asset_cycle_path)
        rotated.rotate(-smallest_idx)
        return list(rotated)

    stringified = rotate_list(
        [str(asset) for asset in asset_cycle_path]
    )  # Ensure stability within a cycle path across runs
    stringified.append(stringified[0])  # Add dependency to first node in cycle
    return " -> ".join(stringified)


def _get_asset_id(path: AssetPath) -> str:
    if path.project_name is None:
        raise LayerClientException("Cannot plan execution with missing project name")
    return path.path()


def _create_not_found_exception(path: AssetPath) -> Exception:
    return ProjectDependencyNotFoundException(
        f"{path.path()} not found. Declare the dependency in your project",
    )
