ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INSTALL_STAMP := .install.stamp
E2E_TEST_HOME := $(ROOT_DIR)/build/e2e-home
TEST_TOKEN_FILE := .test-token
POETRY := $(shell command -v poetry 2> /dev/null)
IN_VENV := $(shell echo $(CONDA_DEFAULT_ENV)$(CONDA_PREFIX)$(VIRTUAL_ENV))
CONDA_ENV_NAME := $(shell echo $(CONDA_DEFAULT_ENV))
UNAME_SYS := $(shell uname -s)
UNAME_ARCH := $(shell uname -m)
REQUIRED_POETRY_VERSION := 1.1.14

.DEFAULT_GOAL:=help

define get_python_package_version
  $(1)==$(shell $(POETRY) show $1 --no-ansi --no-dev | grep version | awk '{print $$3}')
endef

install: check-poetry apple-arm-prereq-install $(INSTALL_STAMP) ## Install dependencies
$(INSTALL_STAMP): pyproject.toml poetry.lock
ifdef IN_VENV
	$(POETRY) install
else
	$(POETRY) install --remove-untracked
endif
	touch $(INSTALL_STAMP)

.PHONY: apple-arm-prereq-install
apple-arm-prereq-install:
ifeq ($(UNAME_SYS), Darwin)
ifeq ($(UNAME_ARCH), arm64)
ifdef CONDA_ENV_NAME
ifeq ($(CONDA_ENV_NAME), base)
	@echo 'Please create a conda environment and make it active'
	@exit 1
else
	@conda install -y $(call get_python_package_version,tokenizers) \
                         $(call get_python_package_version,xgboost) \
                         $(call get_python_package_version,lightgbm) \
                         $(call get_python_package_version,h5py) \
                         $(call get_python_package_version,pyarrow)
endif
else
	@echo 'Not inside a conda environment or conda not installed, this is a requirement for Apple arm processors'
	@echo 'See https://github.com/conda-forge/miniforge/'
	@exit 1
endif
endif
endif

.PHONY: test
test: $(INSTALL_STAMP) ## Run unit tests
	$(POETRY) run pytest test/unit --cov .

test-login: $(TEST_TOKEN_FILE)
$(TEST_TOKEN_FILE):
	@stty -echo
	@printf "Enter test user token for https://app.layer.ai: "
	@read TOKEN && echo $$TOKEN > $(TEST_TOKEN_FILE)
	@stty echo

.PHONY: e2e-test
e2e-test: $(INSTALL_STAMP) $(TEST_TOKEN_FILE)
	@rm -rf $(E2E_TEST_HOME)
ifdef CI
	$(eval DATADOG_ARGS := --ddtrace-patch-all --ddtrace)
endif
	LAYER_DEFAULT_PATH=$(E2E_TEST_HOME) SDK_E2E_TESTS_LOGS_DIR=$(E2E_TEST_HOME)/stdout-logs/ $(POETRY) run python build_scripts/sdk_login.py $(TEST_TOKEN_FILE)
	LAYER_DEFAULT_PATH=$(E2E_TEST_HOME) SDK_E2E_TESTS_LOGS_DIR=$(E2E_TEST_HOME)/stdout-logs/ $(POETRY) run pytest $(E2E_TEST_SELECTOR) -s -n $(E2E_TEST_PARALLELISM) -vv $(DATADOG_ARGS)

.PHONY: format
format: $(INSTALL_STAMP) ## Apply formatters
	$(POETRY) run isort .
	$(POETRY) run black .

.PHONY: lint
lint: $(INSTALL_STAMP) ## Run all linters
	$(POETRY) run isort --check-only .
	$(POETRY) run black --check . --diff
	$(POETRY) run flake8 .
	$(POETRY) run pylint  --recursive yes .
	$(POETRY) run mypy .
	$(POETRY) run bandit -c pyproject.toml -r .

.PHONY: check-poetry
check-poetry:
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. Please install $(REQUIRED_POETRY_VERSION). See https://python-poetry.org/docs/"; exit 2; fi
ifneq ($(shell $(POETRY) --version | awk '{print $$3}'), $(REQUIRED_POETRY_VERSION))
	@echo "Please use Poetry version $(REQUIRED_POETRY_VERSION). Simply run: poetry self update $(REQUIRED_POETRY_VERSION)" && exit 2
endif

.PHONY: check-package-loads
check-package-loads: ## Check that we can load the package without the dev dependencies
	@rm -f $(INSTALL_STAMP)
ifdef IN_VENV
	$(POETRY) install --no-dev
else
	$(POETRY) install --no-dev --remove-untracked
endif
	$(POETRY) run python -c "import layer"

.PHONY: check-colab-violations
check-colab-violations: ## Check that colab pre-installed packages are not clashing with ours
	$(MAKE) -C build_scripts/colab-tests check-colab-violations

.PHONY: check
check: test lint  ## Run test and lint

.PHONY: publish
publish: ## Publish to PyPi - should only run in CI
	@test $${PATCH_VERSION?PATCH_VERSION expected}
	@test $${PYPI_USER?PYPI_USER expected}
	@test $${PYPI_PASSWORD?PYPI_PASSWORD expected}
	$(eval CURRENT_VERSION := $(shell $(POETRY) version --short))
	$(eval PARTIAL_VERSION=$(shell echo $(CURRENT_VERSION) | grep -Po '.*(?=\.)'))
	$(POETRY) version $(PARTIAL_VERSION).$(PATCH_VERSION)
	$(POETRY) publish --build --username $(PYPI_USER) --password $(PYPI_PASSWORD)
	$(POETRY) version $(CURRENT_VERSION)

.PHONY: clean
clean: ## Resets development environment.
	@echo 'cleaning repo...'
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -f .coverage
	@find . -type d -name '*.egg-info' | xargs rm -rf {};
	@find . -depth -type d -name '*.egg-info' -delete
	@rm -rf dist/
	@rm -f $(INSTALL_STAMP)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name "__pycache__" | xargs rm -rf {};
	@echo 'done.'

.PHONY: deepclean
deepclean: clean ## Resets development environment including test credentials and venv
	@rm -rf `poetry env info -p`
	@rm -f $(TEST_TOKEN_FILE)

.PHONY: help
help: ## Show this help message.
	@echo 'usage: make [target]'
	@echo
	@echo 'targets:'
	@grep -E '^[8+a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
