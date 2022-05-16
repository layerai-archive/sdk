# Layer base python repo

Use this as a starting point for standalone python projects

# Development

## Prerequisites
- `pyenv`
- `poetry`
- `make`

## Setup
Run `pyenv install` in the root of this repo to ensure you have the preferred Python version setup

## Makefile
We use `make` as our build system.

Targets:
- `install` - prepares the `poetry` virtual environment. Most of the other tasks will do that automatically for you
- `format` - formats the code
- `test` - runs unit tests
- `lint` - runs linters
- `check` - runs `test` and `lint`
- `publish` - publishes the project to PyPi. This is intended to be used in CI only.
- `clean` - cleans the repo, including the `poetry` virtual environment
- `help` - prints all targets

## Dependency management
The `poetry` documentation about dependency management is [here](https://python-poetry.org/docs/dependency-specification/)

Every time you change dependencies, you should expect a corresponding change to `poetry.lock`. If you use `poetry` directly, it will be done automatically for you. If you manually edit `pyproject.toml`, you need to run `poetry lock` after

### A few tips:
#### How to add a new dependency
```
    poetry add foo
    # or
    poetry add foo=="1.2.3"
```

#### How to add a new dev dependency
```
    poetry add foo --dev
    # or
    poetry add foo=="1.2.3" --dev
```

#### How to get an environment with this package and all dependencies
```
    poetry shell
```

#### How to run something inside the poetry environment
```
    poetry run <...>
```

#### How to update a dependency
```
    poetry update foo
```
