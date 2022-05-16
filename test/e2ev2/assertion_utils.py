from typing import Optional, Union
from uuid import UUID

from layer.api.entity.run_pb2 import Run
from layer.api.ids_pb2 import RunId
from layer.common import LayerClient
from layer.exceptions.exceptions import LayerClientException


class E2ETestAsserter:
    def __init__(self, client: LayerClient):
        self.client = client

    def assert_run_succeeded(self, run_id: Union[UUID, RunId]) -> None:
        run = (
            self._get_run(RunId(value=str(run_id)))
            if isinstance(run_id, UUID)
            else self._get_run(run_id)
        )
        assert run
        assert run.run_status == Run.STATUS_SUCCEEDED

    def assert_run_failed(self, run_id: RunId) -> None:
        run = self._get_run(run_id)
        assert run
        assert run.run_status == Run.STATUS_FAILED

    def _assert_run_status_equals(self, run_id: RunId, run_status: Run.Status) -> None:
        run = self._get_run(run_id)
        assert run
        assert run.run_status == run_status

    def _get_run(self, run_id: RunId) -> Optional[Run]:
        try:
            return self.client.flow_manager.get_run(run_id)
        except LayerClientException:
            return None
