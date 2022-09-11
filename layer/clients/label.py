from typing import Optional, Set
from uuid import UUID

from layerapi.api.ids_pb2 import DatasetBuildId, ModelTrainId
from layerapi.api.service.logged_data.label_api_pb2 import (
    AddLabelsToEntityRequest,
    GetLabelsAttachedToEntityRequest,
    GetLabelsAttachedToEntityResponse,
)
from layerapi.api.service.logged_data.label_api_pb2_grpc import LabelAPIStub

from layer.config import ClientConfig
from layer.utils.grpc.channel import get_grpc_channel


class LabelClient:
    _service: LabelAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "LabelClient":
        client = LabelClient()
        channel = get_grpc_channel(config)
        client._service = LabelAPIStub(channel)  # pylint: disable=protected-access
        return client

    def add_labels_to(
        self,
        label_names: Set[str],
        model_train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        assert (model_train_id and not dataset_build_id) or (
            not model_train_id and dataset_build_id
        )
        if len(label_names) == 0:
            return
        request = AddLabelsToEntityRequest(
            model_train_id=ModelTrainId(value=str(model_train_id))
            if model_train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
            label_name=label_names,
        )
        self._service.AddLabelsToEntity(request=request)

    def get_labels_attached_to(
        self,
        model_train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> Set[str]:
        assert (model_train_id and not dataset_build_id) or (
            not model_train_id and dataset_build_id
        )
        request = GetLabelsAttachedToEntityRequest(
            model_train_id=ModelTrainId(value=str(model_train_id))
            if model_train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
        )
        resp: GetLabelsAttachedToEntityResponse = (
            self._service.GetLabelsAttachedToEntity(request=request)
        )
        return set(map(lambda l: l.name, resp.label))
