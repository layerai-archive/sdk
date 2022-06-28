import pathlib
import re
from collections import defaultdict

from poetry.core.semver.version import Version
from poetry.factory import Factory


# If you add any new entries to this list, please make sure you test your changes in Colab manually first!
IGNORED_VIOLATIONS = {
    "Version check failed for 'click'. Colab has 7.1.2, but 'flask' requires >=8.0",
    "Version check failed for 'cloudpickle'. Colab has 1.3.0, but 'layer' requires >=2.0.0",
    "Version check failed for 'grpcio'. Colab has 1.46.3, but 'grpcio-tools' requires >=1.47.0",
    "Version check failed for 'humanize'. Colab has 0.5.1, but 'layer' requires >=3.11.0",
    "Version check failed for 'itsdangerous'. Colab has 1.1.0, but 'flask' requires >=2.0",
    "Version check failed for 'protobuf'. Colab has 3.17.3, but 'layer-api' requires >=3.19.4",
    "Version check failed for 'pyarrow'. Colab has 6.0.1, but 'layer' requires 7.0.0",
    "Version check failed for 'urllib3'. Colab has 1.24.3, but 'botocore' requires >=1.25.4,<1.27",
}


def parse_colab_packages(path):
    with open(path) as packages_file:
        lines = packages_file.readlines()
    matches = [re.search(r"^([^\s]*) +(\d+\.\d+.*)$", line) for line in lines]

    versions = {
        match.group(1): match.group(2) for match in matches if match is not None
    }

    return versions


def get_package_constraints(path):
    poetry = Factory().create_poetry(path)

    locker = poetry.locker

    constraints = defaultdict(
        list,
        {
            dep.name: [{"dep": dep, "coming_from": "layer"}]
            for dep in poetry.package.requires
        },
    )

    transitive_dependencies = list(
        locker.get_project_dependency_packages(
            project_requires=poetry.package.requires, extras=[]
        )
    )

    for dependency in transitive_dependencies:
        for transitive_dependency in dependency.package.requires:
            constraints[transitive_dependency.name].append(
                {"dep": transitive_dependency, "coming_from": dependency.name}
            )

    return constraints


def find_violations(colab_packages, package_constraints):
    violations = set()
    for pkg, version_str in colab_packages.items():
        version = Version.parse(version_str)
        pkg_constraints = package_constraints[pkg]
        for pkg_constraint in pkg_constraints:
            if not pkg_constraint["dep"].constraint.allows(version):
                violations.add(
                    f"Version check failed for '{pkg}'. Colab has {version_str}, but '{pkg_constraint['coming_from']}' requires {pkg_constraint['dep'].constraint}"
                )
    return violations


if __name__ == "__main__":
    parent_dir = pathlib.Path(__file__).parent
    colab_packages = parse_colab_packages(parent_dir / "colab-packages.txt")
    package_constraints = get_package_constraints(
        parent_dir.parent.resolve().parent.parent
    )
    violations = (
        find_violations(colab_packages, package_constraints) - IGNORED_VIOLATIONS
    )

    if len(violations) > 0:
        print("Found violations:")
        print("\n".join(violations))
        exit(1)
    print("Success")
