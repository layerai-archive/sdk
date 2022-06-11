import pickle
import sys  # nosec: import_pickle
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import cloudpickle


ENTRYPOINT_FILE = Path(__file__).parent / "entrypoint.py"


@dataclass(frozen=True)
class Environment:
    ...


#     # Add requirements to tarball
#     if self.pip_requirements_file:
#         shutil.copy(self.pip_requirements_file, self.environment_path)
#     elif self.pip_packages:
#         with open(self.environment_path, "w") as reqs_file:
#             reqs_file.writelines(
#                 list(map(lambda package: f"{package}\n", self.pip_packages))
#             )


def build_executable_tar(
    path: Path,
    entrypoint: Callable[..., Any],
    resources: Optional[List[Path]] = None,
    poetry_file: Optional[
        Path
    ] = None,  # TODO(volkan) figure out how to save the environment
) -> None:
    curdir = Path()
    with open(path, "w+b") as target_file, tempfile.TemporaryDirectory() as tmp:
        build_directory = Path(tmp)

        # the beginning of the file needs to be the self-extracting header
        target_file.write(HEADER.encode())

        with tarfile.open(fileobj=target_file, mode="w:gz") as tar:
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
export TMPDIR=`mktemp -d /tmp/selfextract.XXXXXX`

ARCHIVE=`awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' $0`

tail -n+$ARCHIVE $0 | tar xz -C $TMPDIR

CDIR=`pwd`
cd $TMPDIR
./entrypoint.py

cd $CDIR
rm -rf $TMPDIR

exit 0

__ARCHIVE_BELOW__
"""
