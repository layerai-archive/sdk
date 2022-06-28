import re
import uuid
from abc import ABCMeta, abstractproperty
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, TypeVar, Union

from yarl import URL

from layer.cache.cache import Cache
from layer.contracts.project_full_name import ProjectFullName
from layer.exceptions.exceptions import LayerClientException


_ASSET_PATH_PATTERN = re.compile(
    r"^(([a-zA-Z0-9_-]+)\/)?(([a-zA-Z0-9_-]+)\/)?(datasets|models)\/([a-zA-Z0-9_-]+)(:([a-z0-9_]*)(\.([0-9]*))?)?(#([a-zA-Z0-9_-]+))?$"
)

BaseAssetType = TypeVar(  # pylint: disable=invalid-name
    "BaseAssetType", bound="BaseAsset"
)


class AssetType(Enum):
    DATASET = "datasets"
    MODEL = "models"


@dataclass(frozen=True)
class AssetPath:
    asset_name: str
    asset_type: AssetType
    # TODO rename to 'account_name'
    org_name: Optional[str] = None
    project_name: Optional[str] = None
    asset_version: Optional[str] = None
    asset_build: Optional[int] = None
    asset_selector: Optional[str] = None

    @classmethod
    def parse(
        cls,
        composite_name: str,
        expected_asset_type: Optional[AssetType] = None,
    ) -> "AssetPath":
        if len(composite_name.split("/")) < 2:
            if not expected_asset_type:
                raise ValueError("Please specify full path or specify asset type")
            composite_name = f"{expected_asset_type.value}/{composite_name}"

        result = _ASSET_PATH_PATTERN.search(composite_name)
        if not result:
            raise ValueError("Asset path does not match expected pattern")
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
                "expected asset type either in the composite name or as argument"
            )
        if asset_type and expected_asset_type and asset_type != expected_asset_type:
            raise ValueError(
                f"expected asset type {expected_asset_type} but found {asset_type}"
            )

        name = groups[5]
        if not name:
            raise ValueError("Asset name missing")
        optional_version = groups[7]
        optional_build = groups[9]
        optional_selector = groups[11]

        return cls(
            asset_name=name,
            asset_type=asset_type,
            asset_version=optional_version,
            asset_build=int(optional_build) if optional_build else None,
            asset_selector=optional_selector,
            project_name=optional_project,
            org_name=optional_org,
        )

    def is_relative(self) -> bool:
        return self.org_name is None or self.project_name is None

    def has_project(self) -> bool:
        return self.project_name is not None and self.project_name != ""

    def path(self) -> str:
        parts = [
            self.org_name,
            self.project_name,
            self.asset_type.value,
            self.asset_name,
        ]
        p = "/".join([part for part in parts if part is not None])
        if self.asset_version is not None:
            p = f"{p}:{self.asset_version}"
            if self.asset_build is not None:
                p = f"{p}.{self.asset_build}"

        if self.asset_selector is not None:
            p = f"{p}#{self.asset_selector}"

        return p

    def with_project_full_name(self, project_full_name: ProjectFullName) -> "AssetPath":
        return replace(
            self,
            project_name=project_full_name.project_name,
            org_name=project_full_name.account_name,
        )

    def url(self, base_url: URL) -> URL:
        if self.org_name is None:
            raise LayerClientException("Account name is required to get URL")
        if self.project_name is None:
            raise LayerClientException("Project name is required to get URL")
        return base_url / self.path()

    def must_org_name(self) -> str:
        if self.org_name is None:
            raise LayerClientException("Account name is required")
        return self.org_name

    def must_project_name(self) -> str:
        if self.project_name is None:
            raise LayerClientException("Project name is required")
        return self.project_name


class BaseAsset(metaclass=ABCMeta):
    def __init__(
        self,
        path: Union[str, AssetPath],
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence["BaseAsset"]] = None,
    ):
        if dependencies is None:
            dependencies = []
        self._path = (
            AssetPath.parse(path, self.asset_type) if isinstance(path, str) else path
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @abstractproperty
    def asset_type(self) -> AssetType:
        ...

    @property
    def name(self) -> str:
        return self._path.asset_name

    @property
    def path(self) -> str:
        return self._path.path()

    @property
    def id(self) -> uuid.UUID:
        assert self._id is not None
        return self._id

    @property
    def dependencies(self) -> Sequence["BaseAsset"]:
        return self._dependencies

    @property
    def project_name(self) -> Optional[str]:
        return self._path.project_name

    def with_id(self: BaseAssetType, id: uuid.UUID) -> BaseAssetType:
        self._id = id
        return self

    def with_project_full_name(self, project_full_name: ProjectFullName) -> "BaseAsset":
        new_path = self._path.with_project_full_name(project_full_name)
        return self.__class__(
            path=new_path,
            id=self.id,
            dependencies=self.dependencies,
        )

    def with_dependencies(
        self: BaseAssetType, dependencies: Sequence["BaseAsset"]
    ) -> BaseAssetType:
        self._dependencies = dependencies
        return self

    def get_cache_dir(self, cache_dir: Optional[Path] = None) -> Optional[Path]:
        cache = Cache(cache_dir=cache_dir).initialise()
        return cache.get_path_entry(str(self.id))

    def is_cached(self, cache_dir: Optional[Path] = None) -> bool:
        return self.get_cache_dir(cache_dir) is not None
