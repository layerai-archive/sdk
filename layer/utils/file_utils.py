import os
import tarfile
from pathlib import Path


def tar_directory(output_filename: str, source_dir: Path) -> None:
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.sep)
