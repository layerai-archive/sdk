from pathlib import Path
from typing import Optional

from layer.config import DEFAULT_LAYER_PATH


class CacheError(Exception):
    pass


class Cache:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = (cache_dir or DEFAULT_LAYER_PATH) / "cache"

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def is_initialised(self) -> bool:
        return self._cache_dir.exists()

    def _ensure_initialised(self) -> None:
        if not self.is_initialised:
            raise CacheError("Cache not initalised")

    def initialise(self) -> "Cache":
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def put_path_entry(self, key: str, path: Path) -> Optional[Path]:
        self._ensure_initialised()
        if not path.exists():
            return None
        abs_path = path.absolute()
        key_path = abs_path.parent / self._cache_dir / key
        abs_path.rename(key_path)
        return key_path

    def get_path_entry(self, key: str) -> Optional[Path]:
        self._ensure_initialised()
        if not key:
            return None
        key_path = self._cache_dir / key
        if key_path.exists():
            return key_path
        return None

    def clear(self) -> None:
        if self._cache_dir.exists():
            import shutil

            shutil.rmtree(self._cache_dir.absolute().as_posix())
