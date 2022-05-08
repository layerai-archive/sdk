import re
from pathlib import Path
from typing import Dict, List

from setuptools import find_packages, setup


ROOT_PATH = Path(__file__).parent.absolute() / "layer"


init_text = (ROOT_PATH / "__init__.py").read_text("utf-8")
try:
    version = re.findall(r'^__version__ = ["\']([^"\']+)["\']\r?$', init_text, re.M)[0]
except IndexError:
    raise RuntimeError("Unable to determine version.")

install_requires = (
    "GitPython==3.1.14",
    "Jinja2>=2.11.3",
    "aiodocker>=0.19.1",
    "aiohttp>=3.7.3,<3.8.0",
    "boto3>=1.16.24",
    "cloudpickle>=2.0.0",
    "cryptography>=3.4.7",
    "grpcio-tools==1.45.0",
    "grpcio==1.45.0",
    "humanize>=3.11.0",
    "idna<3",  # requests==2.25.1 requires idna<3 for some reason
    "jsonschema==3.1.1",
    "mlflow>=1.25.0",
    "networkx>=2.5",
    "packaging<=21.0",  # latest release (21.2) is causing version conflicts with pyparsing
    "pandas>=1.1.2",
    "pickle5~=0.0.11; python_version < '3.8.0'",  # required for pickle5 support in 3.7
    "polling>=0.3.1",
    "prompt_toolkit>=3.0.8",
    "protobuf>=3.12.0",
    "pyarrow==7.0.0",
    "pyjwt>=1.7.1,<2.0.0",
    "rich~=10.12.0",
    "transformers",
    "typing-extensions<4.0.0",  # rich library cannot use 4.0.0 and pip tries to install it
    "validate-email==1.3",
    "yarl>=1.6.3",
    "wrapt>=1.13.3",
    "layer-api==0.9.1",
)

# TODO: leaving extras to conform to Layer `python-project`
extras_require: Dict[str, List[str]] = {
    "test": [],
}

setup(
    name="layer-sdk",
    version=version,
    description="The Layer SDK",
    author="The Layer Team",
    author_email="python-sdk@layer.ai",
    url="https://pypi.org/project/layer-sdk/",
    license="Apache 2",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
)
