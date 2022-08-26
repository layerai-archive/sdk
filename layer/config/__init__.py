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

    if "LAYER_EXECUTABLES" in os.environ and not _str2bool(
        os.environ["LAYER_EXECUTABLES"]
    ):
        return False
    # enabled by default
    return True


def _str2bool(v: str) -> bool:
    return v.lower() not in ("no", "false", "0")
