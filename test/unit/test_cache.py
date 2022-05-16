import random
import string
from pathlib import Path

from layer.cache import Cache


def test_cache_init_default_dir():
    assert Cache().cache_dir == Path.home() / ".layer" / "cache"


def test_cache_init_custom_dir(tmp_path):
    custom_cache_dir = tmp_path / "".join(
        random.choice(string.ascii_lowercase) for _ in range(10)
    )
    assert Cache(cache_dir=custom_cache_dir).cache_dir == custom_cache_dir / "cache"


def test_cache_get_path_entry_returns_none_on_empty_cache(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    assert cache.get_path_entry(key="") is None
    assert cache.get_path_entry(key=None) is None
    assert cache.get_path_entry(key="1") is None


def test_cache_put_path_entry_path_does_not_exist_returns_none(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    file_path = tmp_path / "test_1"
    assert cache.put_path_entry("key_1", file_path) is None


def test_cache_put_path_entry(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    file_path = touch_file(tmp_path / "test_1")
    cached_path = cache.put_path_entry("key_1", file_path)
    assert cached_path == cache.cache_dir / "key_1"
    assert cached_path.exists()
    assert not file_path.exists()


def test_cache_put_and_get_path_entry(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    key = "key_1"
    file_path = touch_file(tmp_path / "path_1")
    cache.put_path_entry(key, file_path)
    assert cache.get_path_entry(key) == cache.cache_dir / key


def test_cache_initialise(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    assert cache.is_initialised


def test_cache_put_same_key_more_than_once(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    path_1 = touch_file(tmp_path / "1")
    path_2 = touch_file(tmp_path / "2")
    cache.put_path_entry("key_1", path_1)
    cache.put_path_entry("key_1", path_2)
    assert cache.get_path_entry("key_1") == cache.cache_dir / "key_1"


def test_cache_clear(tmp_path):
    cache = Cache(cache_dir=tmp_path).initialise()
    assert cache.cache_dir.exists()
    cache.clear()
    assert not cache.cache_dir.exists()


def test_cache_clear_cache_dir_does_not_exist(tmp_path):
    cache = Cache(cache_dir=tmp_path)
    assert not cache.cache_dir.exists()
    cache.clear()
    assert not cache.cache_dir.exists()


def touch_file(path: Path) -> Path:
    path.touch()
    return path
