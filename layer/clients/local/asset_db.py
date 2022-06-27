import json
import os
from enum import Enum
from pathlib import Path


class AssetType(Enum):
    MODEL = 1
    DATASET = 2


class Asset:
    DB = "LayerAsset"

    def __init__(self, path: Path):
        self.path = path

    def get(self, key):
        with open(self.path / self.DB) as file:
            props = json.load(file)
            return props[key]

    def set(self, key, value):
        with open(self.path / self.DB, "r") as file:
            props = json.load(file)

        with open(self.path / self.DB, "w") as file:
            props[key] = value
            file.write(json.dumps(props))


class AssetDB:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = Path(project_name)

    def _get_assets_path(self, asset_type: AssetType):
        return self.project_path / (
            "models" if asset_type == AssetType.MODEL else "datasets"
        )

    def get_asset(self, asset_name: str, asset_id: str, asset_type: AssetType):
        assets_path = self._get_assets_path(asset_type)
        for path in Path(assets_path).rglob(Asset.DB):
            file = open(path)
            asset = json.load(file)
            if asset["id"] == asset_id:
                file.close()
                return Asset(path.parent)

        asset_path = assets_path / asset_name
        version = self.get_last_version(asset_path) + 1
        asset_path = asset_path / str(version)
        os.makedirs(asset_path)
        with open(asset_path / Asset.DB, "w") as file:
            config = {
                "type": asset_type.value,
                "id": asset_id,
                "name": asset_name,
                "project_name": self.project_name,
            }
            file.write(json.dumps(config))
        return Asset(asset_path)

    @staticmethod
    def get_last_version(asset_path: Path):
        if not os.path.exists(asset_path):
            os.makedirs(asset_path)
        dirs = os.listdir(asset_path)
        dirs.sort(key=lambda f: int(f) if f.isnumeric() else -1)
        if len(dirs) == 0:
            return 0
        else:
            return int(dirs[-1])

    @staticmethod
    def get_asset_by_name_and_version(
        project_name: str, asset_name: str, version_id: str, asset_type: AssetType
    ):
        db = AssetDB(project_name)
        path = db._get_assets_path(asset_type) / asset_name
        if version_id is not None:
            if os.path.exists(path / version_id):
                path = path / version_id
                return Asset(path)
        else:
            last_version = AssetDB.get_last_version(path)
            if last_version > 0:
                return Asset(path / str(last_version))

        raise Exception(f"Asset {path} not found!")

    @staticmethod
    def get_asset_by_id(project_name: str, asset_id: str, asset_type: AssetType):
        db = AssetDB(project_name)
        assets_path = db._get_assets_path(asset_type)
        for path in Path(assets_path).rglob(Asset.DB):
            file = open(path)
            asset = json.load(file)
            if asset["id"] == asset_id:
                file.close()
                return Asset(path.parent)
        raise Exception("Asset not found")
