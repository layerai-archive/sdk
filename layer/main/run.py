import logging
from typing import Any, List, Optional

from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.contracts.runs import Run
from layer.executables.runner import remote_run
from layer.global_context import reset_to
from layer.projects.init_project_runner import InitProjectRunner
from layer.projects.utils import validate_project_name
from layer.utils.async_utils import asyncio_run_in_thread

from .utils import sdk_function


logger = logging.getLogger(__name__)


@sdk_function
def init(
    project_name: str,
    fabric: Optional[str] = None,
    pip_packages: Optional[List[str]] = None,
    pip_requirements_file: Optional[str] = None,
) -> Project:
    """
    :param project_name: Name of the project to be initialized.
    :param fabric: Default fabric to use for current project when running code in Layer's backend.
    :param pip_packages: List of packages to install in Layer backend when running code for this project.
    :param pip_requirements_file: File name with list of packages to install in Layer backend when running code for this project.
    :return: Project object.

    Initializes a project with given name both locally and remotely.

    If the project does not exist in the Layer backend, then it will be automatically created.
    Additional settings can be passed in as parameters: namely fabric and packages to install.
    These settings are only applied to runs initiated from the current local environment.

    Other users can set different settings for the same project in their own local environments
    via ``layer.init``, and those will not impact your runs.

    Note that only one of ``pip_packages`` or ``pip_requirements_file`` should be provided.

    .. code-block:: python

        # Initialize new project and create it in Layer backend if it does not exist
        project = layer.init("my_project_name", fabric="x-small")
    """

    validate_project_name(project_name)

    if pip_packages and pip_requirements_file:
        raise ValueError(
            "either pip_requirements_file or pip_packages should be provided, not both"
        )
    layer_config = asyncio_run_in_thread(ConfigManager().refresh())

    project_full_name = _get_project_full_name(layer_config, project_name)

    reset_to(project_full_name)

    init_project_runner = InitProjectRunner(project_full_name, logger=logger)
    fabric_to_set = (
        Fabric(fabric) if fabric else None  # type:ignore # pylint: disable=E1120
    )
    return init_project_runner.setup_project(
        fabric=fabric_to_set,
        pip_packages=pip_packages,
        pip_requirements_file=pip_requirements_file,
    )


@sdk_function
def run(
    functions: List[Any],
    debug: bool = False,
    ray_address: Optional[str] = None,
    **kwargs: Any,
) -> Run:
    """
    :param functions: List of decorated functions to run in the Layer backend.
    :param debug: Stream logs to console from infra executing the project remotely.
    :return: An object with information on the run triggered, unless there is an initialization error.

    Runs the specified functions on Layer infrastructure.

    Running the project does the following:
    - Adds new versions of decorated functions to the Layer web UI.
    - Runs any logs and stores them.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset

        # Define a new function for dataset generation
        @dataset("my_dataset_name")
        def create_my_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe

        # Initialize current project locally and remotely (if it doesn't exist)
        project = layer.init("my_project_name", fabric="x-small")

        # Run the dataset generation code using Layer's backend compute resources
        run = layer.run([create_my_dataset])
        # run = layer.run([create_my_dataset], debug=True)  # Stream logs to console
    """
    _make_notebook_links_open_in_new_tab()

    return remote_run(functions)


def _get_project_full_name(
    layer_config: Config, user_input_project_name: str
) -> ProjectFullName:
    """
    Will first try to extract account_name/project_name from :user_input_project_name

    If no account name can be extracted, will try to get it from global context.

    If that too fails, will fetch the personal account.
    """
    parts = user_input_project_name.split("/")
    account_name: Optional[str]
    project_name: str
    if len(parts) == 1:
        project_name = user_input_project_name
        with LayerClient(layer_config.client, logger).init() as client:
            account_name = client.account.get_my_account().name
    else:
        account_name = parts[0]
        project_name = parts[1]

    return ProjectFullName(
        project_name=project_name,
        account_name=account_name,
    )


# Normally, Colab/IPython opens links as an IFrame. One can open them as new tabs through the right-click menu or using shift+click.
# However, we would like to change the default behavior to always open on a new page.
def _make_notebook_links_open_in_new_tab() -> None:
    def is_notebook() -> bool:
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if is_notebook():
        from IPython.display import Javascript, display  # type: ignore

        # Colab
        display(
            Javascript(
                """
document.querySelectorAll("#output-area a").forEach(function(a){
    a.setAttribute('target', '_blank');
})
        """
            )
        )

        # Jupyter
        display(
            Javascript(
                """
document.querySelectorAll(".output a").forEach(function(a){
    a.setAttribute('target', '_blank');
})
        """
            )
        )
