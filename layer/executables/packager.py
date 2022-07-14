import glob
import inspect
import os
import pickle  # nosec
import shutil  # nosec
import sys
import tempfile
import zipapp
from pathlib import Path
from typing import Any, Callable, List, Optional

from . import cloudpickle


def package_function(
    function: Callable[..., Any],
    resources: Optional[List[Path]] = None,
    pip_dependencies: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Packages layer function as a Python executable."""

    if not inspect.isfunction(function) or function.__name__ == "<lambda>":
        raise ValueError("function must be a function")

    with tempfile.TemporaryDirectory() as source_dir:
        source = Path(source_dir)

        # include cloudpickle itself in the executable
        cloudpickle_source = Path(cloudpickle.__file__).resolve().parent
        cloudpickle_dest = source / "cloudpickle"
        cloudpickle_dest.mkdir(parents=True, exist_ok=True)
        for module_file in glob.iglob(f"{cloudpickle_source}{os.sep}*.py"):
            module_path = Path(module_file)
            shutil.copyfile(module_path, cloudpickle_dest / module_path.name)

        requirements_path = source / "requirements.txt"
        with open(requirements_path, mode="w", encoding="utf8") as requirements:
            requirements.write("\n".join(pip_dependencies or []))

        main_path = source / "__main__.py"
        with open(main_path, mode="w", encoding="utf8") as main:
            main.write(_loader_source())

        if resources:
            _package_resources(source, resources)

        function_path = source / "function"
        with open(function_path, mode="wb") as function_:
            # register to pickle by value to ensure unpickling works anywhere, even if a module is not accessible for the runtime
            cloudpickle.register_pickle_by_value(sys.modules[function.__module__])  # type: ignore
            cloudpickle.dump(function, function_, protocol=pickle.DEFAULT_PROTOCOL)  # type: ignore

        target = (output_dir or Path(".")) / function.__name__

        # create the executable
        zipapp.create_archive(
            source, target, interpreter="/usr/bin/env python", compressed=True
        )

        # ensure the archive is executable
        target.chmod(0o744)

        return target


def _package_resources(source: Path, resources: List[Path]) -> None:
    resource_root = source / "resources"
    resource_root.mkdir(parents=True, exist_ok=True)

    def _copy_resource_file(resource_file: Path) -> None:
        target_path = resource_root / resource_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resource_file, target_path)

    def _walk_dir(dir: Path) -> None:
        for root, dirs, files in os.walk(dir):
            root_path = Path(root)
            for file in files:
                _copy_resource_file(root_path / file)
            for dir_ in dirs:
                _walk_dir(root_path / dir_)

    for resource in resources:
        if resource.is_file():
            _copy_resource_file(resource)
        elif resource.is_dir():
            _walk_dir(resource)


def _loader_source() -> str:
    """Returns the source code of the layer function loader."""

    # entrypoint function body is inlined in the executable's __main__.py
    def _entrypoint() -> None:
        # pylint: disable=W0404
        import os
        import shutil
        import tempfile
        import zipfile
        from pathlib import Path

        import cloudpickle  # type: ignore

        with tempfile.TemporaryDirectory() as env_dir:
            with zipfile.ZipFile(os.path.dirname(__file__)) as exec:
                # extract resources
                for entry in exec.infolist():
                    if entry.filename.startswith("resources/"):
                        with exec.open(entry, mode="r") as resource:
                            target_path = (
                                Path(".") / entry.filename[len("resources/") :]
                            )
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            if not entry.is_dir():
                                with target_path.open(mode="wb") as f:
                                    shutil.copyfileobj(resource, f)

                function_path = os.path.join(env_dir, "function")

                # extract user function
                with exec.open("function", mode="r") as function:
                    with open(function_path, mode="wb") as f:
                        shutil.copyfileobj(function, f)

                # load and run user function
                with open(function_path, mode="rb") as f:
                    cloudpickle.load(f)()

    return f"""
if __name__ == "__main__":
{inspect.getsource(_entrypoint)}
    {_entrypoint.__name__}()
"""
