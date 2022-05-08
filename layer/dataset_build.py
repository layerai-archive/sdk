from uuid import UUID


class DatasetBuild:
    def __init__(
        self,
        dataset_build_id: UUID,
    ):
        self.__dataset_build_id: UUID = dataset_build_id

    def get_id(self) -> UUID:
        return self.__dataset_build_id
