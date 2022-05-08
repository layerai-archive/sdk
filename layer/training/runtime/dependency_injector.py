from logging import Logger
from typing import Any, Callable, Dict

import layer
from layer import Context, Dataset, Model
from layer.annotation_processor import parse_entity_dependencies


def non_optional_current_project_name() -> str:
    project_name = layer.current_project_name()
    assert project_name is not None
    return project_name


def inject_annotated_dependencies(
    func: Callable[..., Any],
    context: Context,
    logger: Logger,
    dataset_factory: Callable[..., Dataset] = layer.get_dataset,
    model_factory: Callable[..., Model] = layer.get_model,
) -> Dict[str, Any]:
    dependency_arguments = parse_entity_dependencies(func)
    logger.debug(f"Entity dependencies: {dependency_arguments.__dict__}")
    injected_params: Dict[str, Any] = {}
    for arg_name in dependency_arguments.raw_datasets:
        raw_dataset = dependency_arguments.raw_datasets[arg_name]
        if not raw_dataset.project_name:
            raw_dataset = raw_dataset.with_project_name(
                non_optional_current_project_name()
            )
        logger.debug(f"Injecting {raw_dataset.path} raw dataset")
        injected_params[arg_name] = dataset_factory(name=raw_dataset.path)
    for arg_name in dependency_arguments.derived_datasets:
        derived_dataset = dependency_arguments.derived_datasets[arg_name]
        if not derived_dataset.project_name:
            derived_dataset = derived_dataset.with_project_name(
                non_optional_current_project_name()
            )
        logger.debug(f"Injecting {derived_dataset.path} derived dataset")
        injected_params[arg_name] = dataset_factory(name=derived_dataset.path)
    for arg_name in dependency_arguments.models:
        model = dependency_arguments.models[arg_name]
        if not model.project_name:
            model = model.with_project_name(non_optional_current_project_name())
        logger.debug(f"Injecting {model.path} model")
        injected_params[arg_name] = model_factory(name=model.path)

    if dependency_arguments.context is not None:
        injected_params[dependency_arguments.context] = context
    if dependency_arguments.train is not None:
        injected_params[dependency_arguments.train] = context.train()
    return injected_params
