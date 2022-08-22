from .config import (  # noqa
    DEFAULT_FUNC_PATH,
    DEFAULT_LAYER_PATH,
    DEFAULT_PATH,
    DEFAULT_URL,
    AuthConfig,
    ClientConfig,
    Config,
    ConfigRecord,
    ConfigStore,
    Credentials,
    LogsConfig,
    S3Config,
)
from .config_client import ConfigClient  # noqa
from .config_manager import ConfigManager  # noqa


def is_executables_feature_active() -> bool:
    import os

    if "LAYER_EXECUTABLES" in os.environ:
        return True

    try:
        config = ConfigManager().load()
    except:  # noqa
        return False

    # enable by default for non production environments
    return config.url != DEFAULT_URL
