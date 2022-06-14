import pickle  # nosec: import_pickle
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional

import cloudpickle


ENTRYPOINT_FILE = Path(__file__).parent / "entrypoint.py"


def build_executable_tar(
    path: Path,
    entrypoint: Callable[..., Any],
    resources: Optional[List[Path]] = None,
    pip_dependencies: Optional[List[str]] = None,
) -> None:

    # add pip dependencies needed by the extractor
    pip_dependencies = pip_dependencies or []
    pip_dependencies.append("cloudpickle")

    curdir = Path()
    with open(path, "w+b") as target_file, tempfile.TemporaryDirectory() as tmp:
        build_directory = Path(tmp)

        # the beginning of the file needs to be the self-extracting header
        target_file.write(HEADER.encode())

        with tarfile.open(fileobj=target_file, mode="w:gz") as tar:
            # Put pip_dependencies as requirements.txt
            requirements_path = build_directory / "requirements.txt"
            with open(requirements_path, mode="w") as file:
                # register to pickly by value to ensure unpickling it works anywhere
                file.write("\n".join(pip_dependencies))
            tar.add(
                requirements_path,
                arcname="requirements.txt",
                filter=_reset_user_info,
            )

            # Put entrypoint.py
            tar.add(
                ENTRYPOINT_FILE,
                arcname="entrypoint.py",
                filter=_reset_user_info,
            )

            # Put pickled function as function.pkl
            function_path = build_directory / "function.pkl"
            with open(function_path, mode="wb") as file:
                # register to pickly by value to ensure unpickling it works anywhere
                cloudpickle.register_pickle_by_value(sys.modules[entrypoint.__module__])
                cloudpickle.dump(entrypoint, file, protocol=pickle.DEFAULT_PROTOCOL)
            tar.add(function_path, arcname="function.pkl", filter=_reset_user_info)

            # Put all resources into the correct path inside the tar
            if resources:
                for resource in resources:
                    tar.add(
                        resource,
                        arcname=resource.relative_to(curdir),
                        filter=_reset_user_info,
                    )

    path.chmod(0o744)


def _reset_user_info(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    return tarinfo


HEADER = """
TMPDIR=`mktemp -d /tmp/selfextract.XXXXXX`
CDIR=`pwd`

function cleanup()
{
    cd $CDIR
    rm -rf $TMPDIR
}

trap cleanup EXIT

# extract contents
ARCHIVE=`awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' $0`
tail -n+$ARCHIVE $0 | tar xz -C $TMPDIR

# run contents in temporary directory
cd $TMPDIR
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python entrypoint.py

exit 0

__ARCHIVE_BELOW__
"""
