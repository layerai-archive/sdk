import json
import re
from typing import Any, Dict, Optional

from layer.exceptions.exceptions import LayerClientException


def _try_parse_grpc_debug_context(message: str) -> Optional[Dict[str, Any]]:
    json_expr = re.search(r"\s({.*?})$", message)
    if json_expr:
        # NOTE: the replace call below is needed to mitigate a bug in gRPC on
        # Windows where backslashes in paths "file":"C:\\..." go unescaped
        json_context = json_expr.group(1).replace("\\", "\\\\")
        try:
            return json.loads(json_context)
        except ValueError:
            return None
    return None


def generate_client_error_from_grpc_error(
    err: Exception, internal_message: str
) -> LayerClientException:
    context = _try_parse_grpc_debug_context(str(err))
    if context:
        grpc_message = context.get("grpc_message")
        if grpc_message is not None:
            return LayerClientException(grpc_message)
    return LayerClientException(f"{internal_message}: {err}")
