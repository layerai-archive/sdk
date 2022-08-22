import filecmp
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from layer.contracts.logged_data import LoggedData, LoggedDataObject, LoggedDataType


logger = logging.getLogger(__name__)


def test_logged_data_object_get_dataframe_successfully() -> None:
    import pandas as pd

    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    as_json = df.to_json(orient="table")
    logged_data = LoggedData(LoggedDataType.TABLE, "tag", data=as_json, epoched_data={})
    logged_data_object = LoggedDataObject(logged_data, epoch=None)
    assert logged_data_object.is_table()
    assert df.equals(logged_data_object.value())


def test_logged_data_object_get_number_successfully() -> None:
    data = "3.14"
    logged_data = LoggedData(LoggedDataType.NUMBER, "tag", data=data, epoched_data={})
    logged_data_object = LoggedDataObject(logged_data, epoch=None)
    assert logged_data_object.is_number()
    assert logged_data_object.value() == float(data)


def test_logged_data_object_get_text_successfully() -> None:
    data = "Some text"
    logged_data = LoggedData(LoggedDataType.TEXT, "tag", data=data, epoched_data={})
    logged_data_object = LoggedDataObject(logged_data, epoch=None)
    assert logged_data_object.is_text()
    assert logged_data_object.value() == data


def test_logged_data_object_get_markdown_successfully() -> None:
    data = "Some text"
    logged_data = LoggedData(LoggedDataType.MARKDOWN, "tag", data=data, epoched_data={})
    logged_data_object = LoggedDataObject(logged_data, epoch=None)
    assert logged_data_object.is_markdown()
    assert logged_data_object.value() == data


def test_logged_data_object_get_boolean_successfully() -> None:
    data = "True"
    logged_data = LoggedData(LoggedDataType.BOOLEAN, "tag", data=data, epoched_data={})
    logged_data_object = LoggedDataObject(logged_data, epoch=None)
    assert logged_data_object.is_boolean()
    assert logged_data_object.value() is True


def test_logged_data_object_get_file_successfully() -> None:
    with patch(
        "requests.get"
    ) as requests_get, tempfile.NamedTemporaryFile() as temp_file1:
        file_data = "some_string test"
        mock_response = MagicMock()
        mock_response.content = file_data.encode("utf-8")
        requests_get.return_value = mock_response
        logged_data = LoggedData(
            LoggedDataType.FILE, "tag", data="url://some", epoched_data={}
        )
        logged_data_object = LoggedDataObject(logged_data, epoch=None)
        logged_data_object.download_to(Path(temp_file1.name))
        assert logged_data_object.is_file()
        with open(temp_file1.name, "r") as f_handle:
            assert file_data == f_handle.read()


def _write_to_file(path: Path, content: str) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(content)


def test_logged_data_object_get_directory_successfully() -> None:
    with patch(
        "requests.get"
    ) as requests_get, tempfile.NamedTemporaryFile() as temp_file1, tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as tmp_dir2:
        _write_to_file(Path(tmp_dir) / "asd" / "asd.txt", "ASD")
        _write_to_file(Path(tmp_dir) / "asd2" / "asd.txt", "ASD2")
        _write_to_file(Path(tmp_dir) / "asd" / "deeper" / "asd22.txt", "ASD3")
        _write_to_file(Path(tmp_dir) / "asd" / "asd3.txt", "ASD4")
        shutil.make_archive(temp_file1.name, "zip", tmp_dir)

        with open(f"{temp_file1.name}.zip", "rb") as f_handle:
            mock_response = MagicMock()
            mock_response.content = f_handle.read()
            requests_get.return_value = mock_response
            logged_data = LoggedData(
                LoggedDataType.DIRECTORY, "tag", data="url://some", epoched_data={}
            )
            logged_data_object = LoggedDataObject(logged_data, epoch=None)
            logged_data_object.download_to(Path(tmp_dir2))
            assert logged_data_object.is_directory()

            dircmp = filecmp.dircmp(tmp_dir, tmp_dir2)
            assert len(dircmp.left_only) == 0
            assert len(dircmp.right_only) == 0
            assert len(dircmp.diff_files) == 0


def test_logged_data_object_get_video_successfully() -> None:
    with patch(
        "requests.get"
    ) as requests_get, tempfile.NamedTemporaryFile() as temp_file1:
        file_data = "some_string_video_data"
        mock_response = MagicMock()
        mock_response.content = file_data.encode("utf-8")
        requests_get.return_value = mock_response
        logged_data = LoggedData(
            LoggedDataType.VIDEO, "tag", data="url://some", epoched_data={}
        )
        logged_data_object = LoggedDataObject(logged_data, epoch=None)
        logged_data_object.download_to(Path(temp_file1.name))
        assert logged_data_object.is_video()
        with open(temp_file1.name, "r") as f_handle:
            assert file_data == f_handle.read()


def test_logged_data_object_get_image_no_epoch_successfully() -> None:
    from PIL import Image

    with patch(
        "requests.get"
    ) as requests_get, tempfile.NamedTemporaryFile() as temp_file1:
        image_data = np.random.rand(400, 400, 3) * 255
        expected_image = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        expected_image.save(temp_file1.name, format="PNG")
        with open(temp_file1.name, "rb") as f_handle:
            mock_response = MagicMock()
            mock_response.content = f_handle.read()
            requests_get.return_value = mock_response
            logged_data = LoggedData(
                LoggedDataType.IMAGE, "tag", data="url://some", epoched_data={}
            )
            logged_data_object = LoggedDataObject(logged_data, epoch=None)
            actual_image = logged_data_object.value()

            assert np.array_equal(np.array(expected_image), np.array(actual_image))


def test_logged_data_object_get_image_with_epoch_successfully() -> None:
    from PIL import Image

    with patch(
        "requests.get"
    ) as requests_get, tempfile.NamedTemporaryFile() as temp_file1:
        image_data = np.random.rand(400, 400, 3) * 255
        expected_image = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
        expected_image.save(temp_file1.name, format="PNG")
        with open(temp_file1.name, "rb") as f_handle:
            mock_response = MagicMock()
            mock_response.content = f_handle.read()
            requests_get.return_value = mock_response
            logged_data = LoggedData(
                LoggedDataType.IMAGE,
                "tag",
                data="url://some",
                epoched_data={3: "url://epoch"},
            )
            logged_data_object = LoggedDataObject(logged_data, epoch=3)
            actual_image = logged_data_object.value()

            requests_get.assert_called_with("url://epoch")
            assert np.array_equal(np.array(expected_image), np.array(actual_image))
