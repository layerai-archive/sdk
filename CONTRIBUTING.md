# Contributing to Layer

Whether you are an experienced open source developer or a first-time contributor, we welcome your contributions to the code and the documentation. Thank you for being part of our community!

### Table Of Contents

[Code of Conduct](#code-of-conduct)

[How can I contribute?](#how-can-i-contribute)
* [Reporting bugs](#reporting-bugs)
* [Suggesting enhancements](#suggesting-enhancements)
* [Contributing code](#contributing-code)

[How do I contribute code?](#how-do-i-contribute-code)
* [Setting up your environment](#setting-up-your-environment)
* [Running in development](#running-in-development)
* [Testing your changes](#testing-your-changes)
* [Submitting a pull request](#submitting-a-pull-request)

## Code of Conduct

This project and everyone participating in it is governed by the [Layer Code of Conduct](https://github.com/layerai/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [community@layer.ai](mailto:community@layer.ai).

## How can I contribute?

### Reporting bugs

This section guides you through submitting a bug report for Layer. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior and find related reports.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report). Fill out [the required template](https://github.com/layerai/.github/blob/master/.github/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check our [discourse](https://discourse.layer.ai/)** for a list of common questions and problems.
* **Perform a [cursory search](https://github.com/search?q=+is%3Aissue+user%3Alayerai)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the relevant repository and fill in [the template](https://github.com/layerai/.github/blob/master/.github/ISSUE_TEMPLATE/bug_report.md).

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened and share more information using the guidelines below.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after updating to a new version of Layer) or was this always a problem?
* If the problem started happening recently, **can you reproduce the problem in an older version of Layer?** What's the most recent version in which the problem doesn't happen? You can install older versions of Layer with `pip install layer==<version>`.
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.

Include details about your configuration and environment:

* **Which version of Layer are you using?** You can get the exact version by running `pip show layer` in your terminal.
* **What's the name and version of the OS you're using**?


### Suggesting enhancements

This section guides you through submitting an enhancement suggestion for Layer, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion and find related suggestions.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion). Fill in [the template](https://github.com/layerai/.github/blob/master/.github/ISSUE_TEMPLATE/feature_request.md), including the steps that you imagine you would take if the feature you're requesting existed.

#### Before Submitting An Enhancement Suggestion

* **Perform a [cursory search](https://github.com/search?q=+is%3Aissue+user%3Alayerai)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the relevant repository and fill in [the template](https://github.com/layerai/.github/blob/master/.github/ISSUE_TEMPLATE/feature_request.md).

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of Layer which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **Explain why this enhancement would be useful** to most Layer users and isn't something that can or should be implemented separately.
* **List some other tools where this enhancement exists.**
* **Specify which version of Layer you're using.** You can get the exact version by running `pip show layer` in your terminal.
* **Specify the name and version of the OS you're using.**


### Contributing code

If you'd like to go beyond reporting bugs and feature requests, you can follow the steps in [How do I contribute code?](#how-do-i-contribute-code) to set up your local development environment and open a pull request yourself. Before you start writing code, we recommend you [search existing issues](https://github.com/search?q=+is%3Aissue+user%3Alayerai) to see if there is someone else already working the feature you'd like to add. If not, please create an issue and mention in the comments that you're planning to open a pull request yourself.

## How do I contribute code?

### Setting up your environment

First, install these prerequisite tools on your development environment:
  - [pyenv](https://github.com/pyenv/pyenv)
  - [poetry](https://python-poetry.org/)
  - [make](https://www.gnu.org/software/make/manual/make.html)

Then, run `pyenv install $(cat .python-version)` in the root of this repo to ensure you have the preferred Python version installed.

### Running in development

#### Makefile
This repo uses `make` as the build system. The following targets can be used throughout your development lifecycle:

- `install` - prepares the `poetry` virtual environment. Most of the other tasks will do that automatically for you
- `format` - formats the code
- `test` - runs unit tests
- `lint` - runs linters
- `check` - runs `test` and `lint`
- `publish` - publishes the project to PyPi. This is intended to be used in CI only.
- `clean` - cleans the repo, including the `poetry` virtual environment
- `help` - prints all targets

### Python setup
We recommend using `pyenv`

Please run `pyenv install $(cat .python-version)` in the root of this repository to setup the recommended python version.

If you are using an M1 machine, we recommend using `conda` via [Miniforge3](https://github.com/conda-forge/miniforge/). Please run

```
# Install Miniforge3 if required
/bin/bash -c "$(curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)"
# Create and activate conda environment
conda create -yq -n sdk python=3.8
conda activate sdk
```

After that you should be able to run the rest of the `make` targets as normal

#### Installation

You can install `layerai/sdk` from the root of this repository with:

```shell
make install
```

Once this step completes successfully, you will be able to run Layer locally by opening a Python prompt from the Poetry virtual environment:

```shell
poetry shell
python
```

And importing Layer from a Python shell within this virtual environment:

```python
import layer
...
```

#### Dependency management

The `poetry` documentation about dependency management is [here](https://python-poetry.org/docs/dependency-specification/)

Every time you change dependencies, you should expect a corresponding change to `poetry.lock`. If you use `poetry` directly, it will be done automatically for you. If you manually edit `pyproject.toml`, you need to run `poetry lock --no-update` after.

#### Poetry tips

Here are a few tips to use poetry:

* How to add a new dependency
  ```shell
  poetry add foo
  # or
  poetry add foo=="1.2.3"
  ```

* How to add a new dev dependency
  ```shell
  poetry add foo --dev
  # or
  poetry add foo=="1.2.3" --dev
  ```

* How to get an environment with this package and all dependencies
  ```shell
  poetry shell
  ```

* How to run something inside the poetry environment
  ```shell
  poetry run <...>
  ```

* How to update a dependency
  ```shell
  poetry update foo
  ```

### Testing your changes

Once you have made your changes and validated them manually, it's important to run automated checks as well. You can do so with the following command:

```shell
make check
```

This will lint your code and run tests defined in `test/`.

If you would like run linting and testing individually, you can also run the following:

```shell
# Runs unit tests
make test
# Runs all linters
make lint
```

#### Unit testing

All unit tests live under `test/unit`. Please add unit tests for any new code that you contribute.

#### E2E tests against the layer platform
All e2e tests live under `test/e2e`. Running these might incur cost so use sparingly

In order to run the  tests, first you need to [create an api key](https://docs.app.layer.ai/docs/getting-started/log-in#log-in-with-an-api-key) from `https://app.layer.ai`. Then run:

```shell
make e2e-test
```

You will be asked for your key which will be stored for subsequent runs in `.test-token`.
You can find the test logs under `build/e2e-home/logs` and also the standard output generated during tests under `build/e2e-home/stdout-logs`.

#### Testing your local SDK build within a Google Colab notebook

1. Run `poetry build`
2. Upload `dist/layer-0.10.0-py3-none-any.whl` to the Colab notebook after a runtime recreation (hint: you can do by `from google.colab import files` and `files.upload()` inside Colab)
3. `pip install layer-0.10.0-py3-none-any.whl`
4. Run the rest of the notebook as normal

#### Linters

This repo uses the following linters:
- [isort](https://pycqa.github.io/isort/)
- [black](https://github.com/psf/black)
- [flake8](https://flake8.pycqa.org/en/latest/)
- [mypy](https://github.com/python/mypy)
- [bandit](https://bandit.readthedocs.io/en/latest/)

Set these up with your IDE to have a smoother development experience and fewer failed checks.

### Submitting a pull request

The final step after developing and testing your changes locally is to submit a pull request and get your contribution merged back into `layerai/sdk`. Please follow the instructions in the GitHub template when creating your PR and fix any status checks that are failing.

To help debug E2E test issues, network logs are captured, zipped and available in the Summary page of each Check GitHub Action page after the E2E tests are executed.

When the PR passes all checks, a `layerai/sdk` maintainer will review your PR. The maintainer may suggest changes to improve code style or clarity, or to add missing tests. When everything is satisfied, the PR can then be merged onto the `main` branch.

That's it! We are looking forward to your contributions!

---

## Credits

Thanks for the [Atom](https://github.com/atom/atom) team for their fantastic open source guidelines which we've adopted in our own guidelines.
