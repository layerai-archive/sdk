from typing import Any

from layer.settings import LayerSettings


def ensure_has_layer_settings(wrapped: Any) -> None:
    if not hasattr(wrapped, "layer"):
        wrapped.layer = LayerSettings()
