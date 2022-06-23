from .config import (  # noqa
    DEFAULT_FUNC_PATH,
    DEFAULT_LAYER_PATH,
    DEFAULT_PATH,
    DEFAULT_URL,
    AccountServiceConfig,
    AuthConfig,
    ClientConfig,
    Config,
    ConfigRecord,
    ConfigStore,
    Credentials,
    DataCatalogConfig,
    FlowManagerServiceConfig,
    LogsConfig,
    ModelCatalogConfig,
    ModelTrainingConfig,
    ProjectServiceConfig,
    S3Config,
    UserLogsServiceConfig,
)
from .config_client import ConfigClient  # noqa
from .config_manager import ConfigManager  # noqa


def is_feature_active(feature_name: str) -> bool:
    import os

    env_var_value = os.environ.get(f"layer_{feature_name}".upper())
    return env_var_value == "1"
