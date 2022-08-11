include include.mk
include environment.mk
include test/colab/colab-test.mk

.DEFAULT_GOAL:=help

define get_python_package_version
  $(1)==$(shell $(POETRY) show $1 --no-ansi --no-dev | grep version | awk '{print $$3}')
endef

define autoreloadpy
from IPython import get_ipython
ipython = get_ipython()

ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
endef
export autoreloadpy

install: $(INSTALL_STAMP) check-poetry ## Install dependencies
$(INSTALL_STAMP): pyproject.toml poetry.lock $(PREREQ_STAMP)
ifdef IN_VENV
	$(POETRY) install
else
	$(POETRY) install --remove-untracked
endif
	@poetry run ipython profile create --ipython-dir=build/ipython
	@echo "$$autoreloadpy" > build/ipython/profile_default/startup/00-autoreload.py
	@touch $(INSTALL_STAMP)

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

.PHONY: colab-test
colab-test: ## Run colab test against image pulled from dockerhub
# Catching sigint/sigterm to forcefully interrupt run on ctrl+c
	@/bin/bash -c "trap \"trap - SIGINT SIGTERM ERR; echo colab-test cancelled by user; exit 1\" SIGINT SIGTERM ERR; $(MAKE) colab-test-internal"

.PHONY: colab-test-local
colab-test-local: ## Run colab test against image built locally
# Catching sigint/sigterm to forcefully interrupt run on ctrl+c
	@/bin/bash -c "trap \"trap - SIGINT SIGTERM ERR; echo colab-test cancelled by user; exit 1\" SIGINT SIGTERM ERR; $(MAKE) colab-test-internal-local"

.PHONY: colab-test-push
colab-test-push: colab-test-build ## Push image built locally to dockerhub
	@docker push $(DOCKER_IMAGE_NAME)

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

.PHONY: deepclean
deepclean: clean ## Resets development environment including test credentials and venv
	@rm -rf `poetry env info -p`
	@rm -rf build
	@rm -f $(TEST_TOKEN_FILE)

.PHONY: help
help: ## Show this help message.
	@echo 'usage: make [target]'
	@echo
	@echo 'targets:'
	@grep --no-filename -E '^[8+a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
