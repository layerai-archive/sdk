from typing import TYPE_CHECKING

import lazy_loader  # noqa

from .context import Context  # noqa
from .contracts.datasets import Dataset  # noqa
from .contracts.logged_data import Image, Markdown, Video  # noqa
from .contracts.models import Model  # noqa
from .contracts.projects import Project  # noqa
from .contracts.runs import Run  # noqa
from .decorators import (  # noqa
    conda,
    dataset,
    fabric,
    model,
    pip_requirements,
    resources,
)
from .decorators.assertions import (  # noqa
    assert_not_null,
    assert_skewness,
    assert_true,
    assert_unique,
    assert_valid_values,
)
from .flavors.custom import CustomModel  # noqa
from .global_context import current_project_full_name  # noqa
from .main.asset import get_dataset, get_model, save_model  # noqa
from .main.auth import (  # noqa
    login,
    login_as_guest,
    login_with_access_token,
    login_with_api_key,
    logout,
    show_api_key,
)
from .main.cache import clear_cache  # noqa
from .main.log import log  # noqa
from .main.run import init, run  # noqa
from .main.version import get_version  # noqa
from .pandas_extensions import Arrays, Images, _register_type_extensions  # noqa


# keep the existing type definitions only for autocompletions in the editors and type checks
if TYPE_CHECKING:
    from .logged_data.callbacks import KerasCallback, XGBoostCallback  # noqa
    from .logged_data.loggers.pytorch_lightning import PytorchLightningLogger  # noqa

# patch __getattr__, __dir__ and __all__ to lazy load the symbols only when they're required
__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "logged_data.callbacks": ["KerasCallback", "XGBoostCallback"],
        "logged_data.loggers.pytorch_lightning": ["PytorchLightningLogger"],
    },
)

_register_type_extensions()

__version__ = get_version()
