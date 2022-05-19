import re
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from yarl import URL

from layer.cache.cache import Cache
from layer.exceptions.exceptions import LayerClientException


_ASSET_PATH_PATTERN = re.compile(
    r"^(([a-zA-Z0-9_-]+)\/)?(([a-zA-Z0-9_-]+)\/)?(datasets|models)\/([a-zA-Z0-9_-]+)(:([a-z0-9_]*)(\.([0-9]*))?)?(#([a-zA-Z0-9_-]+))?$"
)


class AssetType(Enum):
    DATASET = "datasets"
    MODEL = "models"


@dataclass(frozen=True)
class AssetPath:
    entity_name: str
    asset_type: AssetType
    org_name: Optional[str] = None
    project_name: Optional[str] = None
    entity_version: Optional[str] = None
    entity_build: Optional[int] = None
    entity_selector: Optional[str] = None

    @classmethod
    def parse(
        cls,
        composite_name: str,
        expected_asset_type: Optional[AssetType] = None,
    ) -> "AssetPath":
        if len(composite_name.split("/")) < 2:
            if not expected_asset_type:
                raise ValueError("Please specify full path or specify entity type")
            composite_name = f"{expected_asset_type.value}/{composite_name}"

        result = _ASSET_PATH_PATTERN.search(composite_name)
        if not result:
            raise ValueError("Entity path does not match expected pattern")
        groups = result.groups()
        optional_project = groups[3] if groups[3] else groups[1]
        optional_org = groups[1] if groups[3] else None
        maybe_asset_type = groups[4]
        if maybe_asset_type:
            asset_type = AssetType(maybe_asset_type)
        elif expected_asset_type:
            asset_type = expected_asset_type
        else:
            raise ValueError(
                "expected entity type either in the composite name or as argument"
            )
        if asset_type and expected_asset_type and asset_type != expected_asset_type:
            raise ValueError(
                f"expected entity type {expected_asset_type} but found {asset_type}"
            )

        name = groups[5]
        if not name:
            raise ValueError("Entity name missing")
        optional_version = groups[7]
        optional_build = groups[9]
        optional_selector = groups[11]

        return cls(
            entity_name=name,
            asset_type=asset_type,
            entity_version=optional_version,
            entity_build=int(optional_build) if optional_build else None,
            entity_selector=optional_selector,
            project_name=optional_project,
            org_name=optional_org,
        )

    def has_project(self) -> bool:
        return self.project_name is not None and self.project_name != ""

    def path(self) -> str:
        parts = [
            self.org_name,
            self.project_name,
            self.asset_type.value,
            self.entity_name,
        ]
        p = "/".join([part for part in parts if part is not None])
        if self.entity_version is not None:
            p = f"{p}:{self.entity_version}"
            if self.entity_build is not None:
                p = f"{p}.{self.entity_build}"

        if self.entity_selector is not None:
            p = f"{p}#{self.entity_selector}"

        return p

    def with_project_name(self, project_name: str) -> "AssetPath":
        return replace(self, project_name=project_name)

    def with_org_name(self, org_name: str) -> "AssetPath":
        return replace(self, org_name=org_name)

    def url(self, base_url: URL) -> URL:
        if self.org_name is None:
            raise LayerClientException("Account name is required to get URL")
        if self.project_name is None:
            raise LayerClientException("Account name is required to get URL")
        return base_url / self.path()


class BaseAsset:
    def __init__(
        self,
        path: Union[str, AssetPath],
        asset_type: Optional[AssetType] = None,
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence["BaseAsset"]] = None,
    ):
        if dependencies is None:
            dependencies = []
        self._path = (
            AssetPath.parse(path, asset_type) if isinstance(path, str) else path
        )
        self._id = id
        self._dependencies = dependencies

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseAsset):
            return False

        return (
            self._path == other._path
            and self._id == other._id
            and self._dependencies == other._dependencies
        )

    @property
    def name(self) -> str:
        return self._path.entity_name

    @property
    def path(self) -> str:
        return self._path.path()

    def _update_with(self, asset: "BaseAsset") -> None:
        self._path = asset._path  # pylint: disable=protected-access
        self._id = asset.id
        self._dependencies = asset.dependencies

    def _set_id(self, id: uuid.UUID) -> None:
        self._id = id

    @property
    def id(self) -> uuid.UUID:
        assert self._id is not None
        return self._id

    def _set_dependencies(self, dependencies: Sequence["BaseAsset"]) -> None:
        self._dependencies = dependencies

    @property
    def dependencies(self) -> Sequence["BaseAsset"]:
        return self._dependencies

    @property
    def project_name(self) -> Optional[str]:
        return self._path.project_name

    def with_project_name(self, project_name: str) -> "BaseAsset":
        new_path = self._path.with_project_name(project_name=project_name)
        return BaseAsset(
            path=new_path,
            id=self.id,
            dependencies=self.dependencies,
        )

    def get_cache_dir(self, cache_dir: Optional[Path] = None) -> Optional[Path]:
        cache = Cache(cache_dir=cache_dir).initialise()
        return cache.get_path_entry(str(self.id))

    def is_cached(self, cache_dir: Optional[Path] = None) -> bool:
        return self.get_cache_dir(cache_dir) is not None
