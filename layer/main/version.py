import json
import pathlib
import re
import urllib.request

from layer.global_context import has_shown_update_message, set_has_shown_update_message


def get_latest_version() -> str:
    pypi_url = "https://pypi.org/pypi/layer/json"
    response = urllib.request.urlopen(pypi_url).read().decode()  # nosec urllib_urlopen
    data = json.loads(response)

    return data["info"]["version"]


def get_version() -> str:
    with open(
        pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    ) as pyproject:
        text = pyproject.read()
        # Use a simple regex to avoid a dependency on toml
        version_match = re.search(r'version = "(\d+\.\d+\.\d+.*)"', text)

    if version_match is None:
        raise RuntimeError("Failed to parse version")
    return version_match.group(1)


def check_latest_version() -> None:
    if has_shown_update_message():
        return

    latest_version = get_latest_version()
    current_version = get_version()
    if current_version != latest_version:
        print(
            f"You are using the version {current_version} but the latest version is {latest_version}, please upgrade with 'pip install --upgrade layer'"
        )
    set_has_shown_update_message(True)
