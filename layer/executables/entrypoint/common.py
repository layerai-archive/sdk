import os

import layer
from layer.contracts.definitions import FunctionDefinition
from layer.global_context import reset_to, set_has_shown_update_message


ENV_LAYER_API_URL = "LAYER_API_URL"
ENV_LAYER_API_KEY = "LAYER_API_KEY"
ENV_LAYER_API_TOKEN = "LAYER_API_TOKEN"


def initialize(definition: FunctionDefinition) -> None:
    set_has_shown_update_message(True)
    reset_to(definition.project_full_name.path)

    api_url = os.environ.get(ENV_LAYER_API_URL)
    api_key = os.environ.get(ENV_LAYER_API_KEY)
    api_token = os.environ.get(ENV_LAYER_API_TOKEN)

    if api_url:
        if api_key:
            layer.login_with_api_key(api_key, url=api_url)
        elif api_token:
            layer.login_with_access_token(api_key, url=api_token)
