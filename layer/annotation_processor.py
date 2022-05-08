import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union, cast

from .client import Dataset, DerivedDataset, Model, RawDataset, Train
from .context import Context


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityDependencies:
    models: Mapping[str, Model] = field(default_factory=dict)
    raw_datasets: Mapping[str, RawDataset] = field(default_factory=dict)
    derived_datasets: Mapping[str, DerivedDataset] = field(default_factory=dict)
    context: Optional[str] = None
    train: Optional[str] = None  # for backwards compatibility


def parse_entity_dependencies(function: Callable[[], Any]) -> EntityDependencies:
    models: Dict[str, Model] = {}
    raw_datasets: Dict[str, RawDataset] = {}
    derived_datasets: Dict[str, DerivedDataset] = {}
    context: Optional[str] = None
    train: Optional[str] = None
    logger.debug(f"Annotations: {function.__annotations__}")
    for arg, annotation in function.__annotations__.items():
        if isinstance(annotation, Model):
            models[arg] = annotation
        elif isinstance(annotation, DerivedDataset):
            derived_datasets[arg] = annotation
        elif isinstance(annotation, Dataset):
            raw_datasets[arg] = RawDataset(annotation.path)
        elif annotation == Context:
            context = arg
        elif annotation == Train:
            train = arg
        elif arg == "return":
            pass
    return EntityDependencies(
        models=models,
        raw_datasets=raw_datasets,
        derived_datasets=derived_datasets,
        context=context,
        train=train,
    )


def parse_entity_dependencies_from_source_code(
    python_source_content: str,
    function_name: str = "train_model",
) -> EntityDependencies:
    node = ast.parse(python_source_content)
    functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]

    raw_datasets = {}
    derived_datasets = {}
    models = {}
    context: Optional[str] = None
    train: Optional[str] = None

    for functionNode in functions:
        if functionNode.name == function_name:
            for arg in functionNode.args.args:
                if arg.annotation is not None and isinstance(arg.annotation, ast.Call):
                    dependency_type = ""
                    if isinstance(arg.annotation.func, ast.Attribute):
                        dependency_type = arg.annotation.func.attr
                    elif isinstance(arg.annotation.func, ast.Name):
                        dependency_type = arg.annotation.func.id

                    # The ast Str type is deprecated in 3.8 in favour of Const. The latter has a
                    # field called `value`, whereas in Str the same data is accessible via `s`
                    # field. See: https://docs.python.org/3/whatsnew/3.8.html
                    if sys.version_info < (3, 8):
                        str_: ast.Str = cast(Any, arg.annotation.args[0])
                        dependency_path: str = str_.s
                    else:
                        const: ast.Constant = cast(Any, arg.annotation.args[0])
                        dependency_path: str = const.value

                    if dependency_type == "Dataset":
                        raw_datasets[arg.arg] = RawDataset(dependency_path)
                    elif dependency_type == "DerivedDataset":
                        derived_datasets[arg.arg] = DerivedDataset(dependency_path)
                    elif dependency_type == "Model":
                        models[arg.arg] = Model(dependency_path)
                    elif dependency_type == "Context":
                        context = arg.arg
                    elif dependency_type == "Train":
                        train = arg.arg
            break

    return EntityDependencies(
        models=models,
        raw_datasets=raw_datasets,
        derived_datasets=derived_datasets,
        context=context,
        train=train,
    )


def parse_entity_dependencies_from_file(
    path: Path, entrypoint_source_file_name: str, function_name: str
) -> Sequence[Union[Model, DerivedDataset]]:
    with open(path / entrypoint_source_file_name) as file:
        params = parse_entity_dependencies_from_source_code(
            python_source_content=file.read(), function_name=function_name
        )
        return [
            *params.models.values(),
            *params.derived_datasets.values(),
        ]
